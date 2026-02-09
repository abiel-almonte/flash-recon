#pragma once
#include "common/common.h"


Tensor corr_forward(
    const Tensor volume,
    const Tensor coords,
    const int radius
);

Tensor altcorr_forward(
    const Tensor fmap1,
    const Tensor fmap2,
    const Tensor coords,
    const int radius
);
