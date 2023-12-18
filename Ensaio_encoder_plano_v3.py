import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from framework import file_m2k, post_proc, pre_proc
from framework.data_types import ImagingROI
from imaging import tfm
from surface.surface import SurfaceType
from surface.surface import Surface
from parameter_estimation import intsurf_estimation
from parameter_estimation.intsurf_estimation import img_line, img_line_improved
from framework.utils import pointlist_to_cloud_flat
from framework.utils import pcd_to_mesh as p2m
import open3d as o3d

matplotlib.use('TkAgg')

plt.close('all')
n_shots = 99  # 99

surftop = []
surfbot = []
surfflaw = []
pfac1 = []
pfac2 = []
psid1 = []
psid2 = []

# define a resolução ROI - dado em mm
stepx = .1
stepy = .1
stepz = .1

for k in range(n_shots):
    print(k)
    data = file_m2k.read('../../data/Ensaio 1_2.m2k', freq_transd=5, bw_transd=0.7, tp_transd='gaussian',
                         sel_shots= k)
    _ = pre_proc.hilbert_transforms(data)

    # Encontra os pontos da superfície externa
    data.surf = Surface(data, surf_type=SurfaceType.LINE_OLS)
    data.surf.fit()
    waterpath = data.surf.surfaceparam.b

    # Define a ROI para detecção da superfície interna
    height = 25  # Altura da ROI
    width = 30  # Largura da ROI

    corner_roi = np.array([-width / 2, 0.0, (waterpath + 5)])[np.newaxis, :]
    roi = ImagingROI(corner_roi, height=height, width=width, h_len=int(height / stepz), w_len=int(width / stepx),
                     depth=0, d_len=1)

    # pontos da ROI no eixo x
    surfx = roi.w_points  # + pos
    x_encoder = data.encoders_info[32][1]

    # TFM da superfície interna
    tfm.tfm_kernel(data, roi=roi, output_key=k, sel_shot=0, c=5900)
    yt = post_proc.envelope(data.imaging_results[k].image)
    max_col = np.max(yt, axis=0)
    y = yt / yt.max()

    # Improved Surface Estimation
    y = yt / max_col
    a = img_line_improved(y, 0.2)
    z = roi.h_points[a[0].astype(int)]
    w = a[1].reshape(1, (len(z)))
    lamb = 2e-4 #fidelidade
    rho = 0.1
    # print(f'\tEstimating Surface')
    if k > 0:
        bot_ant = bot
        top_ant = top
        x_encoder_ant = x_encoder
    bot, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w, z, lamb, x0=z, rho=rho, eta=.999,
                                                             itmax=250, tol=1e-6, max_iter_cg=1500)
    top = np.interp(surfx, data.surf.x_discr, data.surf.z_discr)

    # Interpolação das falhas
    #plt.figure()
    dist = bot - np.roll(bot, 1)
    for p in range(1, len(dist)):
        signal = np.sign(dist[p])
        if abs(dist[p]) >= 0.3:
            print(p)
            np_y = int((abs(bot[p - 1] - bot[p]) - 0.2) / stepy)
            if bot[p - 1] <= bot[p]:
                flaw_aux = np.linspace(bot[p - 1] - stepy, bot[p] + stepy, np_y)
            else:
                flaw_aux = np.linspace(bot[p - 1] + stepy, bot[p] - stepy, np_y)
            surfflaw.extend([(surfx[p], x_encoder, flaw_aux[i]) for i in range(np_y)])

    # Plano XY
    surftop.extend([(surfx[i], x_encoder, top[i]) for i in range(len(surfx))])
    surfbot.extend([(surfx[i], x_encoder, bot[i]) for i in range(len(surfx))])

    # Plano z
    psid1.extend([(surfx[0], x_encoder, i) for i in (np.linspace(top[0] + stepz, bot[0] - stepz, 30))])
    psid2.extend([(surfx[-1], x_encoder, i) for i in (np.linspace(top[-1] + stepz, bot[-1] - stepz, 30))])

    # # preenchendo o vão  -  Isso o TFM 3D vai fazer
    # if k > 0:
    #     n_points = int((x_encoder-x_encoder_ant)/stepx-2)
    #     x_aux = np.linspace(x_encoder_ant+stepx,x_encoder-stepx,n_points)
    #     top_aux = np.linspace(top_ant, top_, n_points)
    #     bot_aux = np.linspace(bot_ant, bot, n_points)
    #     for l in range (n_points):
    #         # Plano XY
    #         surftop.extend([(surfx_[i], x_aux[l], top_aux[l, i]) for i in range(len(surfx_))])
    #         surfbot.extend([(surfx_[i], x_aux[l], bot_aux[l, i]) for i in range(len(surfx_))])
    #
    #         # Plano z
    #         psid1.extend([(surfx_[0], x_aux[l], i) for i in (np.linspace(top_aux[l,0] + stepz, bot_aux[l,0] - stepz, 30))])
    #         psid2.extend([(surfx_[-1], x_aux[l], i) for i in (np.linspace(top_aux[l,-1] + stepz, bot_aux[l,-1] - stepz, 30))])

    if k == 0:
        for j in range(len(surfx) - 2):
            pfac1.extend(
                [(surfx[j + 1], x_encoder, i) for i in np.linspace(top[j + 1] + stepz, bot[j + 1] - stepz, 30)])


    elif k == n_shots - 1:
        for j in range(len(surfx) - 2):
            pfac2.extend(
                [(surfx[j + 1], x_encoder, i) for i in np.linspace(top[j + 1] + stepz, bot[j + 1] - stepz, 30)])

test_top = np.array(surftop)
test_bot = np.array(surfbot)
test_flaw = np.array(surfflaw)
test_s1 = np.array(psid1)
test_s2 = np.array(psid2)
test_f1 = np.array(pfac1)
test_f2 = np.array(pfac2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(test_top[:, 0], test_top[:, 1], test_top[:, 2])
ax.scatter(test_bot[:, 0], test_bot[:, 1], test_bot[:, 2])
ax.scatter(test_flaw[:, 0], test_flaw[:, 1], test_flaw[:, 2])
ax.scatter(test_s1[:, 0], test_s1[:, 1], test_s1[:, 2])
ax.scatter(test_s2[:, 0], test_s2[:, 1], test_s2[:, 2])
# ax.scatter(test_f1[:, 0], test_f1[:, 1], test_f1[:, 2])
# ax.scatter(test_f2[:, 0], test_f2[:, 1], test_f2[:, 2])

surfbot.extend(surfflaw)
pts = [surftop, surfbot, pfac1, pfac2, psid1, psid2]

steps = (stepx, stepy, stepz)
print(f'Forming Point Cloud with normals')
# Gerar normais e mesh

pcd = pointlist_to_cloud_flat(points=pts, neighbors=30)
#o3d.visualization.draw_geometries([pcd], point_show_normal=True)


print(f'Generating Mesh')
mesh = p2m(pcd, depth=10, smooth=False)
# mesh.compute_triangle_normals()
# o3d.io.write_triangle_mesh('D:/test.stl', mesh, print_progress=True)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=False)
o3d.io.write_triangle_mesh("mesh.stl", mesh)
