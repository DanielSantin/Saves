# -*- coding: utf-8 -*-
"""
Módulo ``utils``
=================

Este módulo contém as funções utilitárias para uso geral do (*framework*).

.. raw:: html

    <hr>

"""
import numpy as np
from framework.data_types import ImagingROI
from framework.post_proc import envelope
from imaging import tfm
import ezdxf
import open3d as o3d


def cmax_tfm(data, corner_roi, height, width, h_res, w_res, shots=None, scat_angle=180):
    dlx = ((data.inspection_params.step_points - data.inspection_params.step_points[0])[:, 0] / w_res).astype(np.int)
    W = int(width / w_res)
    H = int(height / h_res)
    y = np.zeros((H, W))
    width = 2 * np.abs(corner_roi[0, 0])
    W = int(width / w_res)
    if shots is None:
        shots = [0]
    for i in shots:
        roi = ImagingROI(corner_roi, height=height, width=width, h_len=H, w_len=W)
        chave = tfm.tfm_kernel(data, roi=roi, sel_shot=i, scattering_angle=scat_angle, output_key=i)
        image = np.abs(data.imaging_results.pop(chave).image)
        y[:, dlx[i]:dlx[i] + W] = np.fmax(y[:, dlx[i]:dlx[i] + W], image)
    return y


def img_line(image):
    aux = np.argmax(image, 0).astype(np.int32)
    w = np.max(image, 0)
    return aux, w


def pwd_from_fmc(data, theta, xt, c, ts):
    """Gera dados de captura de ondas planas a partir de dados de FMC.

    Parameters
    ----------
    data : :class:`np.ndarray`
        Dados de A-scan, de dimensão :math:`(n_t, n_x, n_x)`, em que
        :math:`n_t` é a quantidade de amostras no tempo e :math:`n_x` é a
        quantidade de elementos do transdutor.

    theta : :class:`np.ndarray`
        Conjunto de ângulos, em radianos, para formar os dados de ondas planas,
        de dimensão :math:`(n_a,)`, em que :math:`n_a` é a quantidade de ângulos.

    xt : :class:`np.ndarray`
        Posições dos transdutores, em mm, de dimensão :math:`(n_x,)`, em que
        :math:`n_x` é a quantidade de elementos do transdutor.
    c : :class:`float`, :class:`int`
        Velocidade de propagação da onda no objeto, em m/s.

    ts : :class:`float`, :class:`int`
        Período de amostragem dos sinais de A-scan, em segundos.

    Returns
    -------
    :class:`np.ndarray`
        Matriz de dimensão :math:`(n_a, n_t, n_x)` com os dados de ondas planas.

    """
    n = data.shape[0]
    m = data.shape[2]
    x = np.zeros((theta.shape[0], n, m), dtype=data.dtype)

    for k, thetak in enumerate(theta):
        for j in range(m):
            tau = xt[j] * np.sin(thetak / 180 * np.pi) / c
            x[k, :, :] += delay_signal(data[:, j, :], tau, ts)

    return x


def delay_signal(x, tau, ts, wrap=False, fill=0):
    r"""Introduz um *delay* de :math:`\tau` no sinal :math:`x`.

    Parameters
    ----------
    x : :class:`np.ndarray`
        Sinal a ser inserido o atraso. Pode ser um vetor de dimensão :math:`(n_x)`
        ou uma matriz de dimensão :math:`(n_y, n_x)`.

    tau : :class:`int`, :class:`float`, :class:`np.ndarray`
        Atraso a ser inserido no sinal. Deve ser um número, caso o sinal :math:`x`
        seja um vetor ou um vetor de dimensão :math:`(n_x)` caso o sinal seja uma
        matriz.

    ts : :class:`int`, :class:`float`
        Período de amostragem do sinal.

    wrap : :class:`bool`
        Define se o atraso é realizado de maneira circular. Se `True`, amostras
        nas extremidades são deslocadas para as extremidades opostas. Por padrão,
        é `True`.

    fill : :class:`int`, :class:`float`
        Se ``wrap`` for `False`, define o valor que o sinal será preenchido onde
        não existir amostras para o deslocamento. Por padrão, é `0`.

    Returns
    -------
    :class:`np.ndarray`
        Sinal com atraso.
    """
    xd = np.empty_like(x)

    # Atraso, em amostras
    n = int(round(tau / ts))

    # Completa com o valor de wrap, se necessário
    if wrap is True:
        if n > 0:
            fill = x[-n:]
        elif n < 0:
            fill = x[:-n]

    # Atrasa o sinal
    if n > 0:
        xd[:n] = fill
        xd[n:] = x[:-n]
    elif n < 0:
        xd[n:] = fill
        xd[:n] = x[-n:]
    else:
        xd = x

    return xd


def save_2d_cad(filename, top, bot):
    doc = ezdxf.new()
    doc.layers.new(name='Top', dxfattribs={'linetype': 'SOLID', 'color': 1})
    doc.layers.new(name='Bottom', dxfattribs={'linetype': 'SOLID', 'color': 3})
    doc.layers.new(name='Sides', dxfattribs={'linetype': 'SOLID', 'color': 5})
    msp = doc.modelspace()
    msp.add_spline_control_frame(top, method='uniform', dxfattribs={'layer': 'Top'})
    msp.add_spline_control_frame(bot, method='uniform', dxfattribs={'layer': 'Bottom'})
    msp.add_line(top[0], bot[0], dxfattribs={'layer': 'Sides'})
    msp.add_line(top[-1], bot[-1], dxfattribs={'layer': 'Sides'})
    doc.saveas(filename)


def pointlist_to_cloud_cylindrical(points, steps, orient_tangent=False, xlen=None, radius_top=10, radius_bot=10):
    stepx, stepy, stepz = steps
    step = np.abs(steps).mean()
    surftop, surfbot, pfac1, pfac2, psid1, psid2 = points

    # Cria objeto PointCloud
    pcdtop = o3d.geometry.PointCloud()
    if surftop:
        # Transforma a lista de pontos em uma estrutura de pontos em 3D
        pcdtop.points = o3d.utility.Vector3dVector(surftop)
        # Usa uma busca em arvore com determinado raio para estimar as normais. Essas normais não podem ser muito dife-
        # rentes, como num spool, para dar problema.
        pcdtop.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_top, max_nn=60))
        if orient_tangent:  # Cria uma consistencia entre as normais. O parametro k define o numero de vizinhos a serem
            # considerados para a consistencia.
            pcdtop.orient_normals_consistent_tangent_plane(k=20)
            # pcdtop.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdtop.normals)))
        elif xlen is not None:  # Altera o alinhamento das direcões para spool
            pcdtop.orient_normals_to_align_with_direction()
            a = np.asarray(pcdtop.normals)
            coef = np.sign(a[:, 1])[:, np.newaxis]
            a *= - coef
            coef = np.repeat(np.sign(a[::xlen][:, 0]), xlen)[:, np.newaxis]
            pcdtop.normals = o3d.utility.Vector3dVector(a * -coef)
        else:  # Top deve ter as normais invertidas para apontar para cima
            pcdtop.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdtop.normals)))

    pcdbot = o3d.geometry.PointCloud()
    if surfbot:
        pcdbot.points = o3d.utility.Vector3dVector(surfbot)
        pcdbot.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_bot, max_nn=60))
        if orient_tangent:  # Cria uma consistencia entre as normais. O parametro k define o numero de vizinhos a serem
            # considerados para a consistencia.
            pcdbot.orient_normals_consistent_tangent_plane(20)
            pcdbot.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdbot.normals)))
        elif xlen is not None:  # Altera o alinhamento das direcões para spool
            pcdbot.orient_normals_to_align_with_direction()
            a = np.asarray(pcdbot.normals)
            coef = np.repeat(np.sign(a[::xlen][:, 0]), xlen)[:, np.newaxis]
            pcdbot.normals = o3d.utility.Vector3dVector(a * coef)
            pcdbot.orient_normals_to_align_with_direction()
            pcdbot.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdbot.normals)))
        else:
            pcdbot.normals = o3d.utility.Vector3dVector((np.asarray(pcdbot.normals)))

    pcdf1 = o3d.geometry.PointCloud()
    if pfac1:
        pcdf1.points = o3d.utility.Vector3dVector(pfac1)
        pcdf1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=step * 2, max_nn=20))
        # pcdf1.orient_normals_to_align_with_direction()
        # pcdf1.normals = o3d.utility.Vector3dVector(-np.asarray(pcdf1.normals)) # Face frontal deve apontar para fora da tela
        pcdf1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, -1, 0])),
                                                           [np.asarray(pcdf1.normals).shape[0], 1]))

    pcdf2 = o3d.geometry.PointCloud()
    if pfac2:
        pcdf2.points = o3d.utility.Vector3dVector(pfac2)
        pcdf2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=step * 2, max_nn=20))
        # pcdf2.orient_normals_to_align_with_direction()
        # pcdf2.normals = o3d.utility.Vector3dVector(np.asarray(pcdf2.normals)) # Face traseira deve apontar para dentro da tela
        pcdf2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, 1, 0])),
                                                           [np.asarray(pcdf2.normals).shape[0], 1]))

    pcds1 = o3d.geometry.PointCloud()
    if psid1:
        pcds1.points = o3d.utility.Vector3dVector(psid1)
        pcds1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=step * 2, max_nn=60))
        pcds1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([-1, 0, 0])),
                                                           [np.asarray(pcds1.normals).shape[0], 1]))
        # Face lateral esquerda deve apontar para esquerda

    pcds2 = o3d.geometry.PointCloud()
    if psid2:
        pcds2.points = o3d.utility.Vector3dVector(psid2)
        pcds2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=step * 2, max_nn=60))
        pcds2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([1, 0, 0])),
                                                           [np.asarray(pcds2.normals).shape[0], 1]))
        # Face lateral direita deve apontar para direita, junto do eixo.

    # Soma as nuvens de pontos em uma única nuvem
    return pcdbot + pcdtop + pcdf1 + pcdf2 + pcds1 + pcds2


def pcd_to_mesh(pcd, depth=7, scale=1.1, smooth=0):
    print('Meshing')
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=scale, linear_fit=True)[0]
    mesh.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
    # mesh.compute_triangle_normals()
    # mesh.compute_vertex_normals()
    if smooth:
        mesh = mesh.filter_smooth_simple(smooth)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
    print(f'Generated mesh with {len(mesh.triangles)} triangles')
    return mesh


def save_2d_cad(filename, top, bot):
    doc = ezdxf.new()
    doc.layers.new(name='Top', dxfattribs={'linetype': 'SOLID', 'color': 1})
    doc.layers.new(name='Bottom', dxfattribs={'linetype': 'SOLID', 'color': 3})
    doc.layers.new(name='Sides', dxfattribs={'linetype': 'SOLID', 'color': 5})
    msp = doc.modelspace()
    msp.add_spline_control_frame(top, method='uniform', dxfattribs={'layer': 'Top'})
    msp.add_spline_control_frame(bot, method='uniform', dxfattribs={'layer': 'Bottom'})
    msp.add_line(top[0], bot[0], dxfattribs={'layer': 'Sides'})
    msp.add_line(top[-1], bot[-1], dxfattribs={'layer': 'Sides'})
    doc.saveas(filename)


def p2c(points: list, outer_surf=True, angle=0, force_direction=np.asarray([0.0, 0.0, 0.0]), radius=10,
        max_neighbours=100, k=80):
    pcd = o3d.geometry.PointCloud()
    if not points:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(points)
    if any(force_direction):
        pcd.normals = o3d.utility.Vector3dVector(np.tile(force_direction, [len(points), 1]))
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbours))
        pcd.orient_normals_consistent_tangent_plane(k=k)
        ref = np.asarray([(2 * outer_surf - 1) * (2 * (np.sin(np.pi * angle / 180) < 0) - 1), 0.0,
                          (2 * outer_surf - 1) * (2 * (np.cos(np.pi * angle / 180) < 0) - 1)])
        if np.cos(np.pi * angle / 180) > 0:
            pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
        # print(ref)
        pcd.orient_normals_to_align_with_direction(ref)
    pcd.normalize_normals()
    return pcd


def pointslist_to_cloud(points: list):
    pcd = o3d.geometry.PointCloud()
    if not points:
        return pcd
    project_vectors = np.asarray([])

    surftop, surfbot, pfac1, pfac2, psid1, psid2 = points

    # Cria objeto PointCloud
    pcdtop = o3d.geometry.PointCloud()
    if surftop:
        # Transforma a lista de pontos em uma estrutura de pontos em 3D
        pcdtop.points = o3d.utility.Vector3dVector(surftop)
        # Usa uma busca em arvore com determinado raio para estimar as normais. Essas normais não podem ser muito dife-
        # rentes, como num spool, para dar problema.
        pcdtop.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_top, max_nn=60))
        if orient_tangent:  # Cria uma consistencia entre as normais. O parametro k define o numero de vizinhos a serem
            # considerados para a consistencia.
            pcdtop.orient_normals_consistent_tangent_plane(k=40)
            pcdtop.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdtop.normals)))
        elif xlen is not None:  # Altera o alinhamento das direcões para spool
            pcdtop.orient_normals_to_align_with_direction()
            a = np.asarray(pcdtop.normals)
            coef = np.sign(a[:, 1])[:, np.newaxis]
            a *= - coef
            coef = np.repeat(np.sign(a[::xlen][:, 0]), xlen)[:, np.newaxis]
            pcdtop.normals = o3d.utility.Vector3dVector(a * -coef)
        else:  # Top deve ter as normais invertidas para apontar para cima
            pcdtop.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdtop.normals)))

    pcdbot = o3d.geometry.PointCloud()
    if surfbot:
        pcdbot.points = o3d.utility.Vector3dVector(surfbot)
        pcdbot.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_bot, max_nn=60))
        if orient_tangent:  # Cria uma consistencia entre as normais. O parametro k define o numero de vizinhos a serem
            # considerados para a consistencia.
            pcdbot.orient_normals_consistent_tangent_plane(40)
            # pcdbot.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdbot.normals)))
        elif xlen is not None:  # Altera o alinhamento das direcões para spool
            pcdbot.orient_normals_to_align_with_direction()
            a = np.asarray(pcdbot.normals)
            coef = np.repeat(np.sign(a[::xlen][:, 0]), xlen)[:, np.newaxis]
            pcdbot.normals = o3d.utility.Vector3dVector(a * coef)
            pcdbot.orient_normals_to_align_with_direction()
            pcdbot.normals = o3d.utility.Vector3dVector((np.asarray(pcdbot.normals)))
        else:
            pcdbot.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdbot.normals)))

    pcdf1 = o3d.geometry.PointCloud()
    if pfac1:
        pcdf1.points = o3d.utility.Vector3dVector(pfac1)
        pcdf1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, -1, 0])),
                                                           [np.asarray(pcdf1.points).shape[0], 1]))

    pcdf2 = o3d.geometry.PointCloud()
    if pfac2:
        pcdf2.points = o3d.utility.Vector3dVector(pfac2)
        pcdf2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, 1, 0])),
                                                           [np.asarray(pcdf2.points).shape[0], 1]))


    pcds1 = o3d.geometry.PointCloud()
    if psid1:
        pcds1.points = o3d.utility.Vector3dVector(psid1)
        pcds1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=step * 2, max_nn=60))
        pcds1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([-1, 0, 0])),
                                                           [np.asarray(pcds1.normals).shape[0], 1]))
        # Face lateral esquerda deve apontar para esquerda

    pcds2 = o3d.geometry.PointCloud()
    if psid2:
        pcds2.points = o3d.utility.Vector3dVector(psid2)
        pcds2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=step * 2, max_nn=60))
        pcds2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([1, 0, 0])),
                                                           [np.asarray(pcds2.normals).shape[0], 1]))
        # Face lateral direita deve apontar para direita, junto do eixo.

    return pcdbot + pcdtop + pcdf1 + pcdf2 + pcds1 + pcds2

#
def pointlist_to_cloud_flat(points, orientNormals=True, neighbors=5, radiusWall=10, radiusTop=10, radiusBot=10, max_nm=60):
    surftop, surfbot, pfac1, pfac2, psid1, psid2 = points


    pcdtop = o3d.geometry.PointCloud()
    pcdtop.points = o3d.utility.Vector3dVector(surftop)
    pcdtop.remove_duplicated_points()
    pcdtop.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, 0, -1])),[np.asarray(pcdtop.points).shape[0], 1]))
    pcdtop.orient_normals_consistent_tangent_plane(k=neighbors)
    pcdtop.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdtop.normals)))


    pcdbot = o3d.geometry.PointCloud()
    pcdbot.points = o3d.utility.Vector3dVector(surfbot)
    pcdbot.remove_duplicated_points()
    pcdbot.estimate_normals()
    #pcdbot.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, 0, 1])),[np.asarray(pcdbot.points).shape[0], 1]))

    pcdbot.orient_normals_consistent_tangent_plane(k=neighbors)

    pcdf1 = o3d.geometry.PointCloud()
    pcdf1.points = o3d.utility.Vector3dVector(pfac1)
    pcdf1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, -1, 0])),[np.asarray(pcdf1.points).shape[0], 1]))

    pcdf2 = o3d.geometry.PointCloud()
    pcdf2.points = o3d.utility.Vector3dVector(pfac2)
    pcdf2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, 1, 0])),[np.asarray(pcdf2.points).shape[0], 1]))

    pcds1 = o3d.geometry.PointCloud()
    pcds1.points = o3d.utility.Vector3dVector(psid1)
    pcds1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([-1, 0, 0])),[np.asarray(pcds1.points).shape[0], 1]))

    pcds2 = o3d.geometry.PointCloud()
    pcds2.points = o3d.utility.Vector3dVector(psid2)
    pcds2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([1, 0, 0])),[np.asarray(pcds2.points).shape[0], 1]))

    # radiusWall = 10
    # radiusBot = 10
    # radiusTop = 10
    # max_nm = 10
    # neighbors = 5

    #pcdwalls.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radiusWall, max_nn=max_nm))

    #pcdwalls.orient_normals_consistent_tangent_plane(k=neighbors)


    #pcd.orient_normals_consistent_tangent_plane(k=neighbors)
    pcd = pcdtop + pcdbot + pcdf1 + pcdf2 + pcds1 + pcds2
    pcd.remove_duplicated_points()
    return pcd

    # def pointlist_to_cloud(points, steps, orient_tangent=False, xlen=None, radius_top=10, radius_bot=10):
    #     stepx, stepy, stepz = steps
    #     step = np.abs(steps).mean()
    #     surftop, surfbot, pfac1, pfac2, psid1, psid2 = points
    #
    #     # Cria objeto PointCloud
    #     pcdtop = o3d.geometry.PointCloud()
    #     if surftop:
    #         # Transforma a lista de pontos em uma estrutura de pontos em 3D
    #         pcdtop.points = o3d.utility.Vector3dVector(surftop)
    #         # Usa uma busca em arvore com determinado raio para estimar as normais. Essas normais não podem ser muito dife-
    #         # rentes, como num spool, para dar problema.
    #         pcdtop.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #         radius=radius_top, max_nn=60))
    #         if orient_tangent: # Cria uma consistencia entre as normais. O parametro k define o numero de vizinhos a serem
    #             # considerados para a consistencia.
    #             pcdtop.orient_normals_consistent_tangent_plane(k=40)
    #             pcdtop.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdtop.normals)))
    #         elif xlen is not None: # Altera o alinhamento das direcões para spool
    #             pcdtop.orient_normals_to_align_with_direction()
    #             a = np.asarray(pcdtop.normals)
    #             coef = np.sign(a[:, 1])[:, np.newaxis]
    #             a *= - coef
    #             coef = np.repeat(np.sign(a[::xlen][:, 0]), xlen)[:, np.newaxis]
    #             pcdtop.normals = o3d.utility.Vector3dVector(a*-coef)
    #         else: # Top deve ter as normais invertidas para apontar para cima
    #             pcdtop.normals = o3d.utility.Vector3dVector((np.asarray(pcdtop.normals)))
    #
    #     pcdbot = o3d.geometry.PointCloud()
    #     if surfbot:
    #         pcdbot.points = o3d.utility.Vector3dVector(surfbot)
    #         pcdbot.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #         radius=radius_bot, max_nn=60))
    #         if orient_tangent: # Cria uma consistencia entre as normais. O parametro k define o numero de vizinhos a serem
    #             # considerados para a consistencia.
    #             pcdbot.orient_normals_consistent_tangent_plane(40)
    #             # pcdbot.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdbot.normals)))
    #         elif xlen is not None: # Altera o alinhamento das direcões para spool
    #             pcdbot.orient_normals_to_align_with_direction()
    #             a = np.asarray(pcdbot.normals)
    #             coef = np.repeat(np.sign(a[::xlen][:, 0]), xlen)[:, np.newaxis]
    #             pcdbot.normals = o3d.utility.Vector3dVector(a*coef)
    #             pcdbot.orient_normals_to_align_with_direction()
    #             pcdbot.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdbot.normals)))
    #         else:
    #             pcdbot.normals = o3d.utility.Vector3dVector((np.asarray(pcdbot.normals)))
    #
    #     pcdf1 = o3d.geometry.PointCloud()
    #     if pfac1:
    #         pcdf1.points = o3d.utility.Vector3dVector(pfac1)
    #         pcdf1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, -1, 0])),
    #                                                            [np.asarray(pcdf1.points).shape[0], 1]))
    #
    #     pcdf2 = o3d.geometry.PointCloud()
    #     if pfac2:
    #         pcdf2.points = o3d.utility.Vector3dVector(pfac2)
    #         pcdf2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([0, 1, 0])),
    #                                                            [np.asarray(pcdf2.points).shape[0], 1]))
    #
    #     pcds1 = o3d.geometry.PointCloud()
    #     if psid1:
    #         pcds1.points = o3d.utility.Vector3dVector(psid1)
    #         pcds1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #         radius=step*2, max_nn=60))
    #         pcds1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([-1, 0, 0])),
    #                                                            [np.asarray(pcds1.normals).shape[0], 1]))
    #         # Face lateral esquerda deve apontar para esquerda
    #
    #     pcds2 = o3d.geometry.PointCloud()
    #     if psid2:
    #         pcds2.points = o3d.utility.Vector3dVector(psid2)
    #         pcds2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #         radius=step*2, max_nn=60))
    #         pcds2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([1, 0, 0])),
    #                                                            [np.asarray(pcds2.normals).shape[0], 1]))
    #         # Face lateral direita deve apontar para direita, junto do eixo.
    #
    #     # Soma as nuvens de pontos em uma única nuvem
    #     return pcdbot + pcdtop + pcdf1 + pcdf2 + pcds1 + pcds2

    # Soma as nuvens de pontos em uma única nuvem
    # return pcdbot + pcdtop + pcdf1 + pcdf2 + pcds1 + pcds2


#

def pcd_to_mesh(pcd, reflexive=True, depth=8, scale=1.1, smooth=0):
    print('Meshing')
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=scale, linear_fit=True)[0]
    mesh.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
    if reflexive:
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
    if smooth:
        mesh = mesh.filter_smooth_simple(smooth)
    print(f'Generated mesh with {len(mesh.triangles)} triangles')
    return mesh


def find_nearest(array, value):
    idx = np.minimum(len(array) - 1, np.searchsorted(array, value, side="left"))
    return idx


def define_waterpath(thickness, cs, target_nb_int_echoes=3, cw=1483, plot=False):
    n_echoes = 0
    wp = 1e-3
    sp = thickness * 1e-3
    fwe = 2 * wp / cw
    swe = 4 * wp / cw
    while not n_echoes == target_nb_int_echoes:
        wp += 1e-3
        fwe = 2 * wp / cw
        swe = 4 * wp / cw
        int_echoes = fwe + 2 * sp / cs * np.arange(1, 100)
        echoes_between2we = int_echoes[int_echoes < swe]
        n_echoes = len(echoes_between2we)
    print(f'Minimum water path required is {wp * 1e3:.2f}mm')
    if plot:
        import matplotlib.pyplot as plt
        int_echoes = fwe + 2 * sp / cs * np.arange(1, 100)
        t = np.linspace(fwe * 0.9, swe * 1.1, 10000)
        echoes = np.zeros_like(t)
        echoes[find_nearest(t, fwe * np.arange(1, 3))] = 5
        int_echoes_idx = find_nearest(t, int_echoes)
        echoes[int_echoes_idx] = 2
        plt.plot(t, echoes)
