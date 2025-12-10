import numpy as np
import open3d as o3d
import torch
import trimesh

def read_mesh_from_path(mesh_path):
    """
    使用 trimesh 读取网格，并转换为 Open3D 格式。
    """
    mesh_trimesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh_trimesh, trimesh.Trimesh):
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
        return mesh_o3d
    else:
        raise ValueError("Unsupported mesh type or scene.")

def mesh_to_o3d(vertices, faces):
    """
    将顶点和面转换为 Open3D TriangleMesh 对象。
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

def o3d_to_mesh(mesh):
    """
    将 Open3D TriangleMesh 对象转换回顶点和面数组。
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return vertices, faces

##### Decimation
def decimate_quadric_edge_collapse(mesh, targetfacenum=None, verbose=False):
    if verbose:
        print(f"Starting decimation... Original faces: {len(mesh.triangles)}")
    if targetfacenum is None:
        targetfacenum = len(mesh.triangles) // 2
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=targetfacenum)
    if verbose:
        print(f"Decimated to {len(mesh.triangles)} faces.")
    return mesh

##### Vertex Merge
def merge_close_vertices(mesh, threshold=0.0001, verbose=False):
    if verbose:
        print(f"Starting merge close vertices... Original vertices: {len(mesh.vertices)}")
    mesh = mesh.remove_duplicated_vertices()
    if verbose:
        print(f"Merged to {len(mesh.vertices)} vertices.")
    return mesh

##### Island Removal
def remove_isolated_pieces(mesh, mincomponentsize=25, verbose=False):
    if verbose:
        print(f"Starting remove isolated pieces... Original faces: {len(mesh.triangles)}")
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    # Open3D 没有直接的连通组件删除，可用 trimesh 辅助
    # 这里先用 trimesh 进行连通组件分析
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    components = tmesh.split(only_watertight=False)
    # 保留最大的连通组件
    main_component = max(components, key=lambda c: len(c.faces))
    mesh = mesh_to_o3d(main_component.vertices, main_component.faces)
    if verbose:
        print(f"Removed isolated pieces. Remaining faces: {len(mesh.triangles)}")
    return mesh

##### Hole Filling
def fix_hole(mesh, maxholesize=30, verbose=False):
    if verbose:
        print("Starting hole filling...")
    # 1. 将 Open3D mesh 转换为 Trimesh 对象
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 2. 使用 Trimesh 的 fill_holes 方法
    # 注意：trimesh 的 fill_holes 会填充所有孔洞，没有直接的 maxholesize 参数。
    # 对于大多数预处理流程，填充所有孔洞是可接受的。
    tmesh.fill_holes()

    # 3. 将修复后的 Trimesh 对象转换回 Open3D mesh
    mesh.vertices = o3d.utility.Vector3dVector(tmesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tmesh.faces)

    if verbose:
        print("Hole filling done.")
    return mesh

##### Repair Non Manifold
def repair_non_manifold(mesh, verbose=False):
    if verbose:
        print("Starting repair non manifold...")
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if verbose:
        print("Non manifold repair done.")
    return mesh

##### Laplacian Smooth
def laplacian_smooth(mesh, stepsmoothnum=3, verbose=False):
    if verbose:
        print(f"Starting Laplacian smoothing for {stepsmoothnum} steps...")
    mesh = mesh.smooth_laplacian(number_of_iterations=stepsmoothnum)
    if verbose:
        print("Laplacian smoothing done.")
    return mesh

##### Taubin Smooth
def taubin_smooth(mesh, stepsmoothnum=3, verbose=False):
    if verbose:
        print(f"Starting Taubin smoothing for {stepsmoothnum} steps...")
    # 使用正确的 Open3D 过滤函数。
    # 注意：这是一个独立的函数，不是 mesh 对象的方法。
    # 它会返回一个新的、经过平滑处理的 mesh 对象。
    mesh = mesh.filter_smooth_taubin(number_of_iterations=stepsmoothnum)
    if verbose:
        print("Taubin smoothing done.")
    return mesh

##### Compute Normal
def compute_normal(mesh, verbose=False):
    if verbose:
        print("Computing vertex normals...")
    mesh.compute_vertex_normals()
    if verbose:
        print("Vertex normals computed.")
    return mesh

##### UV Parameterization
def uv_parameterize_uvatlas(vertices, faces, size=1024, gutter=2.5, max_stretch=0.1666666716337204, parallel_partitions=16, nthreads=0):
    device = o3d.core.Device("CPU:0")
    dtype_f = o3d.core.float32
    dtype_i = o3d.core.int64
    mesh = o3d.t.geometry.TriangleMesh(device)
    mesh.vertex.positions = o3d.core.Tensor(vertices.astype(np.float32), dtype_f, device)
    mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.int64), dtype_i, device)
    mesh.compute_uvatlas(size=size, gutter=gutter, max_stretch=max_stretch, parallel_partitions=parallel_partitions, nthreads=nthreads)
    return mesh.triangle.texture_uvs.numpy()  # (#F, 3, 2)

##### Process Mesh (Open3D version)
def process_mesh(
    vertices,
    faces,
    threshold=0.0001,
    mincomponentRatio=0.02,
    targetfacenum=50000,
    maxholesize=30,
    stepsmoothnum=10,
    verbose=False,
):
    mesh = mesh_to_o3d(vertices, faces)

    # Vertex Merge
    mesh = merge_close_vertices(mesh, threshold=threshold, verbose=verbose)

    # Island Removal
    mesh = remove_isolated_pieces(mesh, mincomponentsize=int(len(faces) * mincomponentRatio), verbose=verbose)

    # Hole Filling
    mesh = repair_non_manifold(mesh, verbose=verbose)
    mesh = fix_hole(mesh, maxholesize=maxholesize, verbose=verbose)

    # Taubin Smoothing
    mesh = taubin_smooth(mesh, stepsmoothnum=stepsmoothnum, verbose=verbose)

    # Decimation if needed
    if len(mesh.triangles) > targetfacenum:
        mesh = decimate_quadric_edge_collapse(mesh, targetfacenum=targetfacenum, verbose=verbose)

    # Final smoothing and normal computation
    mesh = taubin_smooth(mesh, stepsmoothnum=stepsmoothnum, verbose=verbose)
    mesh = repair_non_manifold(mesh, verbose=verbose)
    mesh = compute_normal(mesh, verbose=verbose)

    return o3d_to_mesh(mesh)

##### Process Raw (Open3D version)
def process_raw(mesh_path, save_path, preprocess=True, device="cpu"):
    scene = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    elif isinstance(scene, trimesh.scene.Scene):
        mesh = trimesh.Trimesh()
        for obj in scene.geometry.values():
            mesh = trimesh.util.concatenate([mesh, obj])
    else:
        raise ValueError(f"Unknown mesh type at {mesh_path}.")

    vertices = mesh.vertices
    faces = mesh.faces

    mesh_post_process_options = {
        "mincomponentRatio": 0.02,
        "targetfacenum": 50000,
        "maxholesize": 100,
        "stepsmoothnum": 10,
        "verbose": False,
    }

    if preprocess:
        v_pos, t_pos_idx = process_mesh(
            vertices=vertices,
            faces=faces,
            **mesh_post_process_options,
        )
        mesh_o3d = mesh_to_o3d(v_pos, t_pos_idx)
        mesh_o3d.compute_vertex_normals()
        normals = np.asarray(mesh_o3d.vertex_normals)
    else:
        v_pos, t_pos_idx, normals = vertices, faces, mesh.vertex_normals

    v_tex_np = uv_parameterize_uvatlas(v_pos, t_pos_idx).reshape(-1, 2).astype(np.float32)

    v_pos = torch.from_numpy(v_pos).to(device=device, dtype=torch.float32)
    t_pos_idx = torch.from_numpy(t_pos_idx).to(device=device, dtype=torch.long)
    v_tex = torch.from_numpy(v_tex_np).to(device=device, dtype=torch.float32)
    normals = torch.from_numpy(normals).to(device=device, dtype=torch.float32)

    assert v_tex.shape[0] == t_pos_idx.shape[0] * 3
    t_tex_idx = torch.arange(t_pos_idx.shape[0] * 3, device=device, dtype=torch.long).reshape(-1, 3)

    # De-duplicate UVs (same as your original code)
    v_tex_np_uint32 = v_tex_np.view(np.uint32)
    v_hashed = (v_tex_np_uint32[:, 0].astype(np.uint64) << 32) | v_tex_np_uint32[:, 1].astype(np.uint64)
    v_hashed = torch.from_numpy(v_hashed.view(np.int64)).to(v_pos.device)

    t_pos_idx_f3 = torch.arange(t_pos_idx.shape[0] * 3, device=t_pos_idx.device, dtype=torch.long).reshape(-1, 3)
    v_pos_f3 = v_pos[t_pos_idx].reshape(-1, 3)
    normals_f3 = normals[t_pos_idx].reshape(-1, 3)

    v_hashed_dedup, inverse_indices = torch.unique(v_hashed, return_inverse=True)
    dedup_size, full_size = v_hashed_dedup.shape[0], inverse_indices.shape[0]
    indices = torch.scatter_reduce(
        torch.full([dedup_size], fill_value=full_size, device=inverse_indices.device, dtype=torch.long),
        index=inverse_indices,
        src=torch.arange(full_size, device=inverse_indices.device, dtype=torch.int64),
        dim=0,
        reduce="amin",
    )
    v_tex = v_tex[indices]
    t_tex_idx = inverse_indices.reshape(-1, 3)

    v_pos = v_pos_f3[indices]
    normals = normals_f3[indices]

    normals = normals.to(dtype=torch.float32, device=device)

    uv_to_save = v_tex.clone()
    uv_to_save[:, 1] = 1.0 - uv_to_save[:, 1]

    visual = trimesh.visual.TextureVisuals(uv=uv_to_save.cpu().numpy())
    tmesh = trimesh.Trimesh(
        vertices=v_pos.cpu().numpy(),
        faces=t_tex_idx.cpu().numpy(),
        vertex_normals=normals.cpu().numpy(),
        visual=visual,
        process=False,
    )
    tmesh.export(save_path)
