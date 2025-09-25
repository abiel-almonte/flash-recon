#pragma once

#include <fcntl.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>

#include "buffers.cuh"
#include "preprocessing.hpp"

namespace py = pybind11;

struct CameraFMT{
    v4l2_buf_type type;
    v4l2_field field;
    double fps;
    uint pixelformat;
    uint width; 
    uint height;
    std::string description;
};

inline std::ostream& operator<<(std::ostream& os, const CameraFMT& fmt){
    os << "Camera(width= " << fmt.width << ", height= " << fmt.height << ", fps= " << fmt.fps << ", format= " << fmt.description << ")";
    return os;
}

class CameraFMTList{
    private:
        CameraFMT* data;
        size_t capacity;
        size_t size;
        ssize_t best_idx;

    public:
        
        CameraFMTList() : data(nullptr), best_idx(-1), capacity(0), size(0){}
        ~CameraFMTList() {
            delete[] data;
        }
        
        void append(const CameraFMT& fmt, bool is_best = false){
            if (!data){
                capacity = 1;
                data = new CameraFMT[capacity];

            } else if (size == capacity){
                capacity*=2;
                CameraFMT* new_data = new CameraFMT[capacity];

                for (size_t i = 0; i < size; i++) {
                    new_data[i] = data[i];
                }
                delete[] data;
                data = new_data;
            }

            data[size] = fmt;
            if (is_best){
                best_idx = size;
            }
            size++;
        }

        std::string get_str() {
            std::ostringstream oss;
            for (int i = 0; i < size; i++){
                CameraFMT& fmt = data[i];
                oss << "Camera(" << "id= " << i << ", width= " << fmt.width << ", height= " << fmt.height << ", fps= " << fmt.fps << ", format= " << fmt.description;
                if (i == best_idx){
                    oss << ", BEST";
                }
                oss << ")\n";
            }
            return oss.str();
        }

        const CameraFMT& get_best_fmt() const {
            if (best_idx == -1){
                throw std::out_of_range("No best format");
            }

            return data[best_idx];
        }

        ssize_t get_best_index() const {
            return best_idx;
        }

        const CameraFMT& operator[](int index) const {
            if (index < 0 || index > size - 1){
                throw std::out_of_range("CameraFMTList index out of range");
            }

            return data[index];
        }
};

class Camera{
    private:
        int fd;
        ssize_t fmt_idx;
        CameraFMTList list;
        Preprocessor ppc;
    
    protected:
        CameraRingBuffer h_ring;
        GPURingBuffer d_ring;
        bool is_streaming; 


    public: 
        Camera (const char* filename) : fd(open_camera(filename)), h_ring(fd), d_ring(h_ring.get_n_buffers()), fmt_idx(-1), is_streaming(false){
            check_capabilities();
            fetch_formats();
            set_best_format();
        }

        ~Camera(){
            try {
                stop_streaming();
            } catch(...) {}

            if(fd >= 0){
                close(fd);
            }
        }

        Camera(const Camera&) = delete;
        Camera& operator=(const Camera& other) = delete;

        int open_camera(const char* filename){
            int fd = open(filename, O_RDWR);
            if (fd == -1){
                perror("Failed to open camera");
                throw std::runtime_error("Failed to open camera device");
            }

            return fd;
        }

        void close_camera(){
            stop_streaming();

            if(fd >= 0){
                close(fd);
            }
        }

        friend class FrameGenerator;

        void check_capabilities(){
            v4l2_capability capabilities;
            clear(&capabilities);

            if(ioctl(fd, VIDIOC_QUERYCAP, &capabilities) == -1){
                perror("Failed to get camera capabilites");
                throw std::runtime_error("VIDIOC_QUERYCAP failed");
            } 

            std::cout << "Camera: " << capabilities.card << "\t| Bus: " << capabilities.bus_info << std::endl;

            if (check_for_flag(capabilities.device_caps, V4L2_CAP_STREAMING)){
                std::cout << "  - supports streaming" << std::endl;
            } else{
                std::cout << "  - does NOT support stream" << std::endl;
                throw std::runtime_error("Camera does not support streaming");
            }

            if(check_for_flag(capabilities.device_caps, V4L2_CAP_EXT_PIX_FORMAT)){
                std::cout << "  - supports pixformat" << std::endl;
            } else{
                std::cout << "  - does NOT support pixformat" << std::endl;
                throw std::runtime_error("Camera does not support extended pixel formats");
            }

        }

        void fetch_formats(){
            double max_score = 0;
            for(int desc_idx= 0; ;desc_idx++){
                v4l2_fmtdesc desc;
                clear(&desc);
                desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; //single planar
                desc.index = desc_idx;

                if (ioctl(fd, VIDIOC_ENUM_FMT, &desc) == -1) {
                    if (errno == EINVAL) break;
                    perror("Failed to enumerate formats");
                    break;
                }
                
                if (fmt_is_uncompressed(desc)){
                    for (int res_idx= 0;; res_idx++){
                        v4l2_frmsizeenum res;
                        clear(&res);
                        res.pixel_format = desc.pixelformat;
                        res.index = res_idx;

                        if (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &res) == -1) {
                            if (errno == EINVAL) {
                                break;
                            }
                            perror("Failed to enumerate frame sizes");
                            break;
                        }

                        if (frm_is_discrete(res)){
                            for (int ival_idx = 0; ; ival_idx++) {
                                v4l2_frmivalenum ival;
                                clear(&ival);
                                ival.pixel_format = desc.pixelformat;
                                ival.index = ival_idx;
                                ival.width = res.discrete.width;
                                ival.height = res.discrete.height;
                                
                                if (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &ival) == -1) {
                                    if (errno == EINVAL) {
                                        break;
                                    }
                                    perror("Failed to enumerate frame intervals");
                                    break;
                                }
                                
                                if (frm_ival_is_discrete(ival)) {
                                    double fps = (double)ival.discrete.denominator / ival.discrete.numerator;
                                    double score = fmt_score(fps, res.discrete.width, res.discrete.height);
                                    bool is_best = false;
                                    
                                    CameraFMT fmt = {V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_FIELD_NONE, fps, desc.pixelformat, res.discrete.width, res.discrete.height, std::string((char*)desc.description)};
                                    
                                    if (score > max_score){
                                        max_score = score;
                                        is_best = true;
                                    }

                                    list.append(fmt, is_best);
                                }
                            }
                        }
                    }
                }
            }
        }

        void list_formats(){
            std::cout << list.get_str() << std::endl;
        }

        void set_format(const int index){
            const CameraFMT& curr_fmt = list[index];
            
            v4l2_format format;
            clear(&format);
            
            format.type = curr_fmt.type;
            format.fmt.pix.width = curr_fmt.width;
            format.fmt.pix.height = curr_fmt.height;
            format.fmt.pix.pixelformat = curr_fmt.pixelformat;
            format.fmt.pix.field = curr_fmt.field;
            
            if(ioctl(fd, VIDIOC_S_FMT, &format) == -1){
                perror("Failed to set format");
            }else{
                fmt_idx = index;
                std::cout << "The following camera format has been set:\n  ";
                std::cout << curr_fmt << std::endl;
            }
        }

        void set_best_format(){
            int index = list.get_best_index();
            set_format(index);
        }

        void print_format() const {
            if (fmt_idx == -1) {
                throw std::runtime_error("No format to be retrieved as camera format has not been set");
            }

            const CameraFMT& fmt = list[fmt_idx];
            std::cout << fmt << std::endl;
        }

        void start_streaming(){
            if (is_streaming){
                return;
            }

            h_ring.start_streaming();
            d_ring.init(h_ring);
            is_streaming = true;
        }

        void stop_streaming(){
            if (!is_streaming){
                return;
            }

            h_ring.stop_streaming();
            is_streaming = false;
        }

        torch::Tensor preprocess(int idx){
            const CameraFMT& fmt = list[fmt_idx];

            ppc.process(d_ring[idx], fmt.height, fmt.width);

            torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false);
            return torch::from_blob(d_ring[idx].out, {3, (long)fmt.height, (long)fmt.width}, opts);
        }
};



class FrameGenerator {
    private:
        Camera* cam;

    public:
        FrameGenerator() : cam(nullptr) {}
        FrameGenerator(Camera* _cam ) : cam(_cam) {}

        FrameGenerator* __iter__(){
            return this;
        }

        torch::Tensor __next__(){
            if(!cam->is_streaming){
                cam->start_streaming();
            }

            int idx = cam->h_ring.dequeue_buffer();

            if (idx == -1){
                throw py::stop_iteration();
            }

            cam->d_ring.copy_buffer_to_gpu(idx, cam->h_ring.buffer_start(idx));
            cam->h_ring.queue_buffer(idx);
            
            return cam->preprocess(idx);
        }
};
