
#include <array>
#include "fusion.h"

void Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera, float* pointX)
{
    float tmpX[3];
    // Reprojection
    pointX[0] = depth * (x - camera.K[2]) / camera.K[0];
    pointX[1] = depth * (y - camera.K[3]) / camera.K[1];
    pointX[2] = depth;

    // Rotation
    tmpX[0] = camera.R[0] * pointX[0] + camera.R[3] * pointX[1] + camera.R[6] * pointX[2];
    tmpX[1] = camera.R[1] * pointX[0] + camera.R[4] * pointX[1] + camera.R[7] * pointX[2];
    tmpX[2] = camera.R[2] * pointX[0] + camera.R[5] * pointX[1] + camera.R[8] * pointX[2];

    // Transformation
    float C[3];
    C[0] = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C[1] = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C[2] = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    
    pointX[0] = tmpX[0] + C[0];
    pointX[1] = tmpX[1] + C[1];
    pointX[2] = tmpX[2] + C[2];
}


void ProjectonCamera(const float *PointX, const Camera camera, float *point, float &depth)
{
    float tmp[3];
    tmp[0] = camera.R[0] * PointX[0] + camera.R[1] * PointX[1] + camera.R[2] * PointX[2] + camera.t[0];
    tmp[1] = camera.R[3] * PointX[0] + camera.R[4] * PointX[1] + camera.R[5] * PointX[2] + camera.t[1];
    tmp[2] = camera.R[6] * PointX[0] + camera.R[7] * PointX[1] + camera.R[8] * PointX[2] + camera.t[2];

    depth = tmp[2];
    point[0] = (camera.K[0] * tmp[0] + camera.K[2] * tmp[2]) / depth;
    point[1] = (camera.K[1] * tmp[1] + camera.K[3] * tmp[2]) / depth;
}

float GetAngle( const float *v1, const float *v2)
{
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle = acosf(dot_product);
    //if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if ( angle != angle )
        return 0.0f;

    return angle;
}


void showProgress(int progress, int total) {
    float percentage = static_cast<float>(progress) / total;
    int barWidth = 100;
    int progressWidth = barWidth * percentage;

    std::cout << "Fusion: |";
    for (int i = 0; i < progressWidth; ++i) {
        std::cout << "\u2588";
    }
    for (int i = progressWidth; i < barWidth; ++i) {
        std::cout << " ";
    }
    std::cout << "| " << std::setw(3) << int(percentage * 100.0) << "%\r";
    std::cout.flush();

    if (progress == total) {
        std::cout << std::endl;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> multi_view_fusion(
    const std::vector<torch::Tensor> &images_list,
    const std::vector<torch::Tensor> &Kvec,
    const std::vector<torch::Tensor> &Rmat,
    const std::vector<torch::Tensor> &tvec,
    const std::vector<torch::Tensor> &depths_list,
    const std::vector<torch::Tensor> &normals_list,
    const std::vector<torch::Tensor> &masks_list,
    const std::vector<std::vector<int>> &pairs_list,
    const int min_num_consistency,
    const float reproj_error_threshold,
    const float relative_depth_threshold,
    const float normal_threshold,
    const int step
) {
    size_t N = images_list.size();
    if (N <= 0)
        throw std::runtime_error("Image list size is 0.");
    
    std::vector<Camera> cameras(N, Camera());
    std::vector<float*> images(N, nullptr);
    std::vector<float*> depths(N, nullptr);
    std::vector<float*> normals(N, nullptr);
    std::vector<float*> masks(N, nullptr);

    for (size_t i = 0; i < N; ++i) {

        CHECK_CONTIGUOUS(images_list[i]);
        CHECK_CONTIGUOUS(Kvec[i]);
        CHECK_CONTIGUOUS(Rmat[i]);
        CHECK_CONTIGUOUS(tvec[i]);
        CHECK_CONTIGUOUS(depths_list[i]);
        CHECK_CONTIGUOUS(normals_list[i]);

        size_t C = images_list[i].size(0);
        size_t H = images_list[i].size(1);
        size_t W = images_list[i].size(2);
        size_t HW = H * W;
        size_t CHW = C * HW;

        memcpy(&(cameras[i].K), Kvec[i].cpu().contiguous().data_ptr<float>(), sizeof(float) * 4);
        memcpy(&(cameras[i].R), Rmat[i].cpu().contiguous().data_ptr<float>(), sizeof(float) * 9);
        memcpy(&(cameras[i].t), tvec[i].cpu().contiguous().data_ptr<float>(), sizeof(float) * 3);
        cameras[i].width = W; cameras[i].height = H;
        
        images[i] = new float[CHW];
        memcpy(images[i], images_list[i].cpu().contiguous().data_ptr<float>(), sizeof(float) * CHW);

        depths[i] = new float[HW];
        memcpy(depths[i], depths_list[i].cpu().contiguous().data_ptr<float>(), sizeof(float) * HW);

        normals[i] = new float[3 * HW];
        memcpy(normals[i], normals_list[i].cpu().contiguous().data_ptr<float>(), sizeof(float) * 3 * HW);

        masks[i] = new float[HW];
        torch::Tensor mask_i;
        if(masks_list.size() > 0){
            CHECK_CONTIGUOUS(masks_list[i]);
            mask_i = masks_list[i].cpu().contiguous();
        }
        else{
            mask_i = torch::ones_like(depths_list[i].cpu());
        }

        memcpy(masks[i], mask_i.data_ptr<float>(), sizeof(float) * HW);
    }

    std::vector<std::array<float, 3>> points_position;
    std::vector<std::array<float, 3>> points_color;
    std::vector<std::array<float, 3>> points_normal;

    points_position.clear();
    points_color.clear();
    points_normal.clear();

    for (size_t i = 0; i < N; ++i) {
        showProgress(i, N - 1);

        int num_ngb = pairs_list[i].size();
        std::vector<std::vector<int>> used_list(num_ngb, std::vector<int>({-1, -1}));
        
        for (int r = 0; r < cameras[i].height; r+=step) {
            for (int c = 0; c < cameras[i].width; c+=step) {

                int ref_pix_idx = r * cameras[i].width + c;

                if (masks[i][ref_pix_idx] < 0.5f)
                    continue;
                
                float ref_depth = depths[i][ref_pix_idx];
                float ref_normal[3] = {
                    normals[i][ref_pix_idx], 
                    normals[i][ref_pix_idx + cameras[i].width * cameras[i].height], 
                    normals[i][ref_pix_idx + 2 * cameras[i].width * cameras[i].height]
                };

                if (ref_depth < 1e-10f)
                    continue;

                float PointX[3];
                Get3DPointonWorld(c, r, ref_depth, cameras[i], PointX); // TODO: check +0.5?

                float consistent_Point[3] = {PointX[0], PointX[1], PointX[2]};
                float consistent_normal[3] = {ref_normal[0], ref_normal[1], ref_normal[2]};
                float consistent_Color[3] = {
                    images[i][ref_pix_idx], 
                    images[i][ref_pix_idx + cameras[i].width * cameras[i].height], 
                    images[i][ref_pix_idx + 2 * cameras[i].width * cameras[i].height]
                };

                int ok = 0;

                for (int j = 0; j < num_ngb; ++j) {

                    int src_id = pairs_list[i][j];

                    float point[2];
                    float proj_depth;
                    ProjectonCamera(PointX, cameras[src_id], point, proj_depth);

                    int src_r = int(point[1] + 0.5f);
                    int src_c = int(point[0] + 0.5f);
                    
                    if (src_c >= 0 && src_c < cameras[src_id].width && src_r >= 0 && src_r < cameras[src_id].height) {
                        
                        int src_pix_idx = src_r * cameras[src_id].width + src_c;

                        if (masks[src_id][src_pix_idx] < 0.5f)
                            continue;

                        float src_depth = depths[src_id][src_pix_idx];
                        float src_normal[3] = {
                            normals[src_id][src_pix_idx], 
                            normals[src_id][src_pix_idx + cameras[src_id].width * cameras[src_id].height], 
                            normals[src_id][src_pix_idx + 2 * cameras[src_id].width * cameras[src_id].height]
                        };
                        if (src_depth < 1e-10f)
                            continue;

                        float tmp_X[3]; 
                        Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id], tmp_X);
                        float tmp_pt[2];
                        ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);

                        float reproj_error = sqrt(pow(c - tmp_pt[0], 2) + pow(r - tmp_pt[1], 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);

                        if (reproj_error < reproj_error_threshold && 
                            relative_depth_diff < relative_depth_threshold && 
                            angle < normal_threshold * DEGREE_TO_RADIAN){

                            consistent_Point[0] += tmp_X[0];
                            consistent_Point[1] += tmp_X[1];
                            consistent_Point[2] += tmp_X[2];
                            
                            consistent_normal[0] += src_normal[0];
                            consistent_normal[1] += src_normal[1];
                            consistent_normal[2] += src_normal[2];

                            consistent_Color[0] += images[src_id][src_pix_idx];
                            consistent_Color[1] += images[src_id][cameras[src_id].width * cameras[src_id].height + src_pix_idx];
                            consistent_Color[2] += images[src_id][2 * cameras[src_id].width * cameras[src_id].height + src_pix_idx];

                            used_list[j][0] = src_c;
                            used_list[j][1] = src_r;
                            ok++;
                        }
                    }
                }

                if (ok >= min_num_consistency) {
                    consistent_Point[0] /= (ok + 1.0f);
                    consistent_Point[1] /= (ok + 1.0f);
                    consistent_Point[2] /= (ok + 1.0f);

                    consistent_normal[0] /= (ok + 1.0f);
                    consistent_normal[1] /= (ok + 1.0f);
                    consistent_normal[2] /= (ok + 1.0f);
                    
                    consistent_Color[0] /= (ok + 1.0f);
                    consistent_Color[1] /= (ok + 1.0f);
                    consistent_Color[2] /= (ok + 1.0f);

                    points_position.push_back({consistent_Point[0], consistent_Point[1], consistent_Point[2]});
                    points_color.push_back({consistent_Color[0], consistent_Color[1], consistent_Color[2]});
                    points_normal.push_back({consistent_normal[0], consistent_normal[1], consistent_normal[2]});

                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j][0] == -1)
                            continue;
                        
                        int used_idx = used_list[j][1] * cameras[pairs_list[i][j]].width + used_list[j][0];
                        masks[pairs_list[i][j]][used_idx] = 0.f;
                    }
                }
            }
        }
    }

    // From PointCloud to Tensor
    // std::string ply_path = "patch_match.ply";
    // StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
    auto float_opts = images_list[0].options().dtype(torch::kFloat32).device(torch::kCPU);
    int size = points_position.size();
    torch::Tensor xyz = torch::empty({size, 3}, float_opts);
    torch::Tensor rgb = torch::empty({size, 3}, float_opts);
    torch::Tensor nxyz = torch::empty({size, 3}, float_opts);

    std::memcpy(xyz.data_ptr<float>(), points_position.data(), size * 3 * sizeof(float));
    std::memcpy(rgb.data_ptr<float>(), points_color.data(), size * 3 * sizeof(float));
    std::memcpy(nxyz.data_ptr<float>(), points_normal.data(), size * 3 * sizeof(float));
    
    // free memory
    for (size_t i = 0; i < N; ++i){
        delete[] images[i];
        delete[] depths[i];
        delete[] normals[i];
        delete[] masks[i];
        
        images.clear();
        depths.clear();
        normals.clear();
        masks.clear();

        points_position.clear();
        points_color.clear();
        points_normal.clear();
    }

    return std::make_tuple(xyz, rgb, nxyz);
}
