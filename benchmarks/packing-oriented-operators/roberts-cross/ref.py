import torch


def torch_kernel(img: torch.Tensor) -> torch.Tensor:
    rows, cols = img.shape
    gx_kernel = [[1, 0], [0, -1]]
    gy_kernel = [[0, 1], [-1, 0]]
    output = torch.zeros(rows, cols)
    for x in range(rows):
        for y in range(cols):
            gx = 0.0
            gy = 0.0
            for i in range(2):
                for j in range(2):
                    ni = x + i
                    nj = y + j
                    val = img[ni, nj].item() if ni < rows and nj < cols else 0.0
                    gx += gx_kernel[i][j] * val
                    gy += gy_kernel[i][j] * val
            output[x, y] = gx * gx + gy * gy
    return output
