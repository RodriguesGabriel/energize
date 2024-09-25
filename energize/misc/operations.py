import torch
from torch import nn


class Operations:
    def __init__(self, device: str):
        self.device = device

    def fix_dimensions_matrix(self, a, b):
        c = a.flatten()
        d = b.flatten()
        if len(c) > len(d):
            d = nn.functional.pad(d, (0, len(c) - len(d)), "constant", 1)
            final_shape = a.shape
        else:
            c = nn.functional.pad(c, (0, len(d) - len(c)), "constant", 1)
            final_shape = b.shape

        return c.reshape(final_shape), d.reshape(final_shape)

    def default_tensor(self):
        return torch.Tensor([1]).to(self.device)

    def fix(self, a, b=None):
        a = self.default_tensor() if a is None else torch.where(
            torch.isnan(a), torch.tensor(1.0), a)
        b = self.default_tensor() if b is None else torch.where(
            torch.isnan(b), torch.tensor(1.0), b)
        return a, b

    def sum(self, a, b):
        a, b = self.fix(a, b)
        try:
            return (a + b)
        except RuntimeError as e:
            c, d = self.fix_dimensions_matrix(a, b)
            return (c + d)

    def element_wise_product(self, a, b):
        a, b = self.fix(a, b)
        try:
            return a * b
        except RuntimeError as e:
            c, d = self.fix_dimensions_matrix(a, b)
            return c * d

    def mat_mul(self, a, b):
        a, b = self.fix(a, b)
        if len(a.shape) == 0:
            a = a.unsqueeze(0)
        if len(b.shape) == 0:
            b = b.unsqueeze(0)
        try:
            return a @ b
        except RuntimeError as e:
            c, d = self.fix_dimensions_matrix(a, b)
            return c @ d

    def less_than(self, a, b):
        a, b = self.fix(a, b)
        try:
            return (a < b).to(self.device).float()
        except RuntimeError as e:
            c, d = self.fix_dimensions_matrix(a, b)
            return (c < d).to(self.device).float()

    def greater_than(self, a, b):
        a, b = self.fix(a, b)
        try:
            return (a > b).to(self.device).float()
        except RuntimeError as e:
            c, d = self.fix_dimensions_matrix(a, b)
            return (c > d).to(self.device).float()

    def equal(self, a, b):
        a, b = self.fix(a, b)
        try:
            return (a == b).to(self.device).float()
        except RuntimeError as e:
            c, d = self.fix_dimensions_matrix(a, b)
            return (c == d).to(self.device).float()

    def add_noise(self, a):
        a, _ = self.fix(a)
        return a + (0.1**0.5)*torch.randn(a.shape).to(self.device)

    def log(self, a):
        a, _ = self.fix(a)
        a[a <= 0] = 1
        return torch.log(a)

    def abs(self, a):
        a, _ = self.fix(a)
        return torch.abs(a)

    def power(self, a):
        a, _ = self.fix(a)
        return torch.pow(a, 2)

    def exp(self, a):
        a, _ = self.fix(a)
        return torch.exp(a)

    def normalize(self, a):
        a, _ = self.fix(a)
        mean, std = a.mean(), a.std()
        z = (a - mean)/std
        z[z != z] = 0
        return z

    def relu(self, a):
        a, _ = self.fix(a)
        return torch.functional.F.relu(a, inplace=False)

    def sign(self, a):
        a, _ = self.fix(a)
        return torch.sign(a)

    def heaviside(self, a):
        a, _ = self.fix(a)
        return torch.heaviside(a, values=torch.Tensor([0]).to(self.device).float())

    def element_wise_invert(self, a):
        a, _ = self.fix(a)
        result = 1 / a
        result[result != result] = 0
        return result

    def gaussian_init(self, a):
        a, _ = self.fix(a)
        return torch.normal(mean=0, std=1, size=a.shape).to(self.device)

    def frobenius_norm(self, a):
        a, _ = self.fix(a)
        return torch.norm(a.float(), p='fro')

    def determinant(self, a):
        a, _ = self.fix(a)
        if len(a.shape) < 3 or a.shape[-1] != a.shape[-2]:
            return a

        # Get the size of the square matrix
        matrix_size = a.shape[-1]

        # Flatten the leading dimensions
        leading_dims = a.shape[:-2]
        flattened_tensor = a.reshape(-1, matrix_size, matrix_size)

        # Compute the determinant for each square sub-matrix

        determinants = torch.linalg.det(flattened_tensor)
        result = determinants.reshape(*leading_dims)

        return result

    def logdet(self, a):
        try:

            a, _ = self.fix(a)
            z = torch.logdet(a)
            z[z != z] = 0
            return z
        except RuntimeError as e:
            return a

    def symmetrized_eigenvalues_first_last_ratio(self, a):
        a, _ = self.fix(a)
        a = a + a.t()
        eigenvalues = torch.linalg.eigvalsh(a)
        return eigenvalues[0] / eigenvalues[-1]

    def square_tensor_eigenvalues_first_last_ratio(self, a):
        a, _ = self.fix(a)
        if len(a.shape) != 0:
            lsize = a.shape[0]
            a = a.reshape(lsize, -1)
            a = torch.einsum('nc,mc->nm', [a, a])
            e = torch.linalg.eigvals(a)
            return (e[-1] / e[0])[0]
        return torch.Tensor([0]).to(self.device)

    def normalized_sum(self, a):
        a, _ = self.fix(a)
        return torch.sum(a) / a.numel()

    def l1_norm(self, a):
        a, _ = self.fix(a)
        return torch.sum(abs(a)) / a.numel()

    def hamming_dist(self, a, b):
        a, b = self.fix(a, b)
        a_bin = torch.heaviside(
            a.flatten(), values=torch.Tensor([0]).to(self.device).float())
        b_bin = torch.heaviside(
            b.flatten(), values=torch.Tensor([0]).to(self.device).float())
        return sum(a_bin != b_bin).float()

    def kl_div(self, a, b):
        a, b = self.fix(a, b)

        try:
            z = torch.nn.KLDivLoss(reduction='batchmean')(a, b).float()
        except RuntimeError as e:
            c, d = self.fix_dimensions_matrix(a, b)
            z = torch.nn.KLDivLoss(reduction='batchmean')(c, d).float()
        if z.isnan().any():
            return self.default_tensor()
        return z

    def cosine_similarity(self, a, b):
        a, b = self.fix(a, b)
        try:
            return torch.nn.functional.cosine_similarity(a, b, dim=0).float()
        except RuntimeError as e:
            c, d = self.fix_dimensions_matrix(a, b)
            return torch.nn.functional.cosine_similarity(c, d, dim=0).float()

    def softmax(self, a):
        a, _ = self.fix(a)
        return torch.functional.F.softmax(a)

    def sigmoid(self, a):
        a, _ = self.fix(a)
        return torch.functional.F.sigmoid(a)

    def ones_like(self, a):
        a, _ = self.fix(a)
        return torch.ones_like(a)

    def zeros_like(self, a):
        a, _ = self.fix(a)
        return torch.zeros_like(a)

    def greater_than_zero(self, a):
        a, _ = self.fix(a)
        return (a > 0).float()

    def less_than_zero(self, a):
        a, _ = self.fix(a)
        return (a < 0).float()

    def numel(self, a):
        a, _ = self.fix(a)
        return torch.Tensor([a.numel()]).to(self.device)

    def subtract(self, a, b):
        a, b = self.fix(a, b)
        c, d = self.fix_dimensions_matrix(a, b)
        return c-d

    def transpose(self, a):
        a, _ = self.fix(a)
        try:
            return a.t()
        except RuntimeError:
            return a

    def max(self, a, b):
        a, b = self.fix(a, b)
        # return torch.min(a, b)
        try:
            return torch.max(a, b)
        except RuntimeError as e:
            flat_a = a.flatten()
            flat_b = b.flatten()
            # Find the length of the smaller tensor
            min_length = min(flat_a.size(0), flat_b.size(0))
            # Truncate both tensors to the size of the smaller tensor
            truncated_a = flat_a[: min_length]
            truncated_b = flat_b[: min_length]
            z = torch.max(truncated_a, truncated_b)
            if a.numel() == min_length:
                return z.reshape(a.shape)
            else:
                return z.reshape(b.shape)

    def min(self, a, b):
        a, b = self.fix(a, b)
        # return torch.min(a, b)
        try:
            return torch.min(a, b)
        except RuntimeError as e:
            flat_a = a.flatten()
            flat_b = b.flatten()
            # Find the length of the smaller tensor
            min_length = min(flat_a.size(0), flat_b.size(0))
            # Truncate both tensors to the size of the smaller tensor
            truncated_a = flat_a[: min_length]
            truncated_b = flat_b[: min_length]
            z = torch.min(truncated_a, truncated_b)
            if a.numel() == min_length:
                return z.reshape(a.shape)
            else:
                return z.reshape(b.shape)
