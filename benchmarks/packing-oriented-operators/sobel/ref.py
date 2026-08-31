import torch


def torch_kernel(img: torch.Tensor) -> torch.Tensor:
    rows, cols = img.shape
    gx_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    gy_kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    output = torch.zeros(rows, cols)
    for x in range(rows):
        for y in range(cols):
            gx = 0.0
            gy = 0.0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    ni = x + i
                    nj = y + j
                    if 0 <= ni < rows and 0 <= nj < cols:
                        val = img[ni, nj].item()
                        gx += gx_kernel[i + 1][j + 1] * val
                        gy += gy_kernel[i + 1][j + 1] * val
            output[x, y] = gx * gx + gy * gy
    return output
