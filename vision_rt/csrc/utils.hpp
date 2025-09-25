#pragma once

#include <linux/videodev2.h>
#include <stdint.h>
#include <unistd.h>

#include <cmath>

template<typename T>
inline void clear(T* buffer){
    memset(buffer, 0, sizeof(T));
}

inline bool check_for_flag(uint32_t payload, uint32_t flag){
    if((payload & flag) == 0){
        return false;
    }
    
    return true;
}

inline bool fmt_is_uncompressed(v4l2_fmtdesc& desc){
    return !check_for_flag(desc.flags, V4L2_FMT_FLAG_COMPRESSED);
}

inline bool frm_is_discrete(v4l2_frmsizeenum& res){
    return res.type == V4L2_FRMSIZE_TYPE_DISCRETE;
}

inline bool frm_ival_is_discrete(v4l2_frmivalenum& ival){
    return ival.type == V4L2_FRMIVAL_TYPE_DISCRETE;
}

// score = alpha * log( sqrt(w * h) ) + beta * log(fps)
inline double fmt_score(double fps, int w, int h, double alpha = 1.5, double beta = 1.5) {
    if (fps <= 0 || w <= 0 || h <= 0) {
        return -INFINITY;
    }

    const double lin = std::sqrt(static_cast<double>(w) * static_cast<double>(h));
    return alpha * std::log(lin) + beta * std::log(fps);
}
