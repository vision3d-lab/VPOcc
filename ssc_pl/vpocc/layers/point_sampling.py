import torch
from tqdm import tqdm

class Vanishing_point_based_point_sampling:
    def __init__(self, grid_size, num_heads, num_levels, num_points, beta=50, c=[1, 1.5, 2]):
        self.batch_size = 3000
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.beta = beta
        self.c = torch.tensor(c)
        self.scale_list = self.beta * self.c

        self.grid_size = torch.tensor([grid_size[1], grid_size[0]]).float() # (w, h)
        self.offset = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]]).float()
        self.sampling_locations_batch = torch.empty((1, self.batch_size, num_heads, num_levels, num_points, 2))
        
    def find_line_intersection(self, point_1_1, point_1_2, point_2_1, point_2_2):
        slope_1 = (point_1_2[..., 1] - point_1_1[..., 1]) / (point_1_2[..., 0] - point_1_1[..., 0])
        slope_2 = (point_2_2[..., 1] - point_2_1[..., 1]) / (point_2_2[..., 0] - point_2_1[..., 0])

        intercept_1 = point_1_1[..., 1] - slope_1 * point_1_1[..., 0]
        intercept_2 = point_2_1[..., 1] - slope_2 * point_2_1[..., 0]

        x_intersection = (intercept_2 - intercept_1) / (slope_1 - slope_2)
        y_intersection = slope_1 * x_intersection + intercept_1

        return torch.stack([x_intersection, y_intersection], dim=-1)
    
    def calculate_perpendicular_intersection(self, rotated_points, vanishing_point, index1, index2):
        point_on_line1 = rotated_points[:,:,index1,:]
        point_on_line2 = rotated_points[:,:,index2,:]

        line1_direction = vanishing_point.squeeze(0) - point_on_line1
        line2_direction = vanishing_point.squeeze(0) - point_on_line2
        perpendicular_line2_direction = torch.stack([line2_direction[..., 1], -line2_direction[..., 0]], dim=-1)
        
        return self.line_intersection((point_on_line1, line1_direction), (point_on_line2, perpendicular_line2_direction))

    def line_intersection(self, line1, line2):
        p1, d1 = line1
        p2, d2 = line2
        A = torch.stack([d1, -d2], dim=-1)
        b = p2 - p1
        t = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
        return p1 + d1 * t.squeeze(-1)

    def calculate_sampling_locations(self, ref_pix, vanishing_point, num_initial_offset=4):
        vanishing_point = vanishing_point.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        self.grid_size = self.grid_size.to(ref_pix.device)
        self.offset = self.offset.to(ref_pix.device)
        self.scale_list = self.scale_list.to(ref_pix.device)
        
        unnorm_ref_pix = ref_pix * self.grid_size
        device = ref_pix.device
        num_pixels = ref_pix.size(1)
        num_batches = (num_pixels + self.batch_size - 1) // self.batch_size
        
        distance_ref_to_vp = torch.clamp(torch.norm(vanishing_point - unnorm_ref_pix, dim=-1, keepdim=True), min=1.0)
        total_sampling_locations = torch.empty((1, num_pixels, self.num_heads, self.num_levels, self.num_points, 2), device=device)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_pixels)
            ref_pix_batch = unnorm_ref_pix[:, start_idx:end_idx, ...]
            sampling_locations_batch = torch.empty((1, end_idx - start_idx, self.num_heads, self.num_levels, self.num_points, 2), device=device)
            sampling_locations_batch[:,:,:,:,0,:,] = ref_pix_batch.unsqueeze(2).expand(-1, -1, self.num_heads, -1, -1)

            for i in range(self.num_levels):
                scale = torch.clamp(distance_ref_to_vp[:, start_idx:end_idx, i, :] * 0.5, max=self.scale_list[i])
                offsets = self.offset.reshape(1, 1, 4, 2) * scale.unsqueeze(2).expand(-1,-1,-1,2)
                ref_pix_expanded = ref_pix_batch[:,:,i,:].unsqueeze(2).expand(-1, -1, num_initial_offset, -1)
                sampled_points = ref_pix_expanded + offsets

                vector_to_vp = vanishing_point - ref_pix_expanded
                angle_to_x_axis = torch.atan2(vector_to_vp[..., 1], vector_to_vp[..., 0])
                rotation_matrix = torch.stack([
                    torch.cos(-angle_to_x_axis), -torch.sin(-angle_to_x_axis),
                    torch.sin(-angle_to_x_axis), torch.cos(-angle_to_x_axis)
                ], dim=-1).reshape(1, -1, 4, 2, 2)
                
                rotated_points = torch.matmul((sampled_points - ref_pix_expanded).unsqueeze(3), rotation_matrix).reshape(ref_pix_batch[:,:,i,:].unsqueeze(2).expand(-1, -1, offsets.shape[2], -1).shape) \
                    + ref_pix_expanded

                sampling_locations_batch[:, :, :, i, 1:5, :] = rotated_points.unsqueeze(2).expand(-1, -1, self.num_heads,  -1, -1)

                for j, (n, m) in enumerate([(1, 2), (3, 0), (1, 0), (3, 2)]):

                    if j ==0:
                        intersection_points = self.calculate_perpendicular_intersection(
                        rotated_points, 
                        vanishing_point,
                        n,  # index1
                        m   # index2
                    ) 
                        intersection_points_0 = intersection_points
                    elif j == 1:
                        intersection_points = self.calculate_perpendicular_intersection(
                        rotated_points,
                        vanishing_point,
                        n,  # index1
                        m   # index2
                    ) 
                        intersection_points_1 = intersection_points
                    elif j == 2:
                        intersection_points = self.find_line_intersection(intersection_points_0, rotated_points[:,:,1,:], intersection_points_1, rotated_points[:,:,0,:])
                    elif j == 3:
                        intersection_points = self.find_line_intersection(intersection_points_0, rotated_points[:,:,2,:], intersection_points_1, rotated_points[:,:,3,:])

                    sampling_locations_batch[:, :, :, i, j+5, :] = intersection_points.unsqueeze(2).expand(-1,-1,self.num_heads,-1)

            total_sampling_locations[:, start_idx:end_idx, :, :, :, :] = sampling_locations_batch
        
        total_sampling_locations[..., 0:1] = torch.clamp(total_sampling_locations[..., 0:1], min=0.0, max=self.grid_size[0])
        total_sampling_locations[..., 1:2] = torch.clamp(total_sampling_locations[..., 1:2], min=0.0, max=self.grid_size[1])
        
        sampling_locations = total_sampling_locations / self.grid_size
        
        return  sampling_locations