import torch


def torch_kernel(img: torch.Tensor) -> torch.Tensor:
    rows, cols = img.shape
    kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    output = torch.zeros(rows, cols)
    for x in range(rows):
        for y in range(cols):
            t = 0.0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    ni = x + i
                    nj = y + j
                    if 0 <= ni < rows and 0 <= nj < cols:
                        t += kernel[i + 1][j + 1] * img[ni, nj].item()
            output[x, y] = t
    return output
