#include <cuda_runtime.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <stdexcept>

namespace py = pybind11;


class GraphCachedModule{
    private:
        cudaStream_t stream{nullptr};
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t instance{nullptr};

        py::function fn;
        torch::Tensor in;
        torch::Tensor out;

        int warmup_steps;
        bool graph_created;

    public:

        GraphCachedModule(py::object pytorch_module) : warmup_steps(20), graph_created(false) {
            pytorch_module.attr("compile")();
            fn = pytorch_module.attr("forward");
            
            cudaError_t err = cudaStreamCreate(&stream);
            if (err != cudaSuccess){
                std::string msg = "Failed to create capture stream for GraphCachedModule: ";
                msg += cudaGetErrorString(err);
                throw std::runtime_error(msg);
            }
        }

        ~GraphCachedModule() {
            if (instance){
                cudaGraphExecDestroy(instance);
            }
            if (graph){
                cudaGraphDestroy(graph);
            }
            if (stream){
                cudaStreamDestroy(stream);
            }
        }

        void capture(const torch::Tensor& tensor) {
            in = torch::empty_like(tensor, tensor.options());

            for (int i = 0; i < warmup_steps; i ++){
                fn(tensor);
            }
            cudaDeviceSynchronize();

            torch::Tensor out_tensor = fn(tensor).cast<torch::Tensor>();
            out = torch::empty_like(out_tensor, out_tensor.options());

            cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            if (err != cudaSuccess){
                std::string msg = "Failed to begin graph capture for GraphCachedModule: ";
                msg += cudaGetErrorString(err);
                throw std::runtime_error(msg);
            }

            {
                c10::cuda::CUDAStream capture_stream = c10::cuda::getStreamFromExternal(stream, 0);
                c10::cuda::CUDAStreamGuard guard(capture_stream);
                
                out.copy_(fn(in).cast<torch::Tensor>());
            }

            err = cudaStreamEndCapture(stream, &graph);
            if (err != cudaSuccess){
                std::string msg = "Failed to end graph capture for GraphCachedModule: ";
                msg += cudaGetErrorString(err);
                throw std::runtime_error(msg);
            }

            err = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            if (err != cudaSuccess){
                std::string msg = "Failed to instantiate graph for GraphCachedModule: ";
                msg += cudaGetErrorString(err);
                throw std::runtime_error(msg);
            }
            
            cudaStreamSynchronize(stream);
            
            graph_created = true;
        }

        bool is_captured() const {
            return graph_created;
        }

        torch::Tensor __call__(const torch::Tensor& tensor){
            if(!graph_created){
                throw std::runtime_error("Graph must be captured first with .capture");
            }

            in.copy_(tensor);
            
            cudaGraphLaunch(instance, stream);
            cudaStreamSynchronize(stream);
            return out;
        }
};