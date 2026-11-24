"""Preserve the teddy surface while replacing skinny contact triangles.

The original surface and texture stay available. This creates a new asset;
it does not alter material, attach vertices, or move simulation state.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pymeshlab
import tetgen
import trimesh
from assets_builder import bear_uv,write_obj

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--edge",type=float,default=.009)
    parser.add_argument("--name",default="bear_uniform")
    args=parser.parse_args()
    assets=HERE/"assets"
    ms=pymeshlab.MeshSet()
    ms.load_new_mesh(str(assets/"bear_physics.obj"))
    ms.meshing_isotropic_explicit_remeshing(iterations=8,
        targetlen=pymeshlab.PureValue(args.edge),featuredeg=50.,
        checksurfdist=True,maxsurfdist=pymeshlab.PureValue(.0007))
    current=ms.current_mesh()
    v,f=current.vertex_matrix(),current.face_matrix()
    mesh=trimesh.Trimesh(v,f,process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError("Remeshing changed closed teddy topology")
    write_obj(assets/(args.name+".obj"),v,f,bear_uv(v),"teddy_fur.png")
    points,tets,*_=tetgen.TetGen(v,f).tetrahedralize(switches="pq1.3/15a4e-6YQ")
    tri=points[tets]
    volume=np.abs(np.linalg.det(tri[:,1:]-tri[:,:1]))/6
    edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    quality=12*(3*volume)**(2/3)/np.sum(
        np.linalg.norm(tri[:,edges[:,0]]-tri[:,edges[:,1]],axis=2)**2,axis=1)
    report={"method":"isotropic surface remeshing and boundary-preserving TetGen",
        "target_surface_edge_m":args.edge,"maximum_requested_surface_deviation_m":.0007,
        "points":len(points),"tets":len(tets),"min_volume":float(volume.min()),
        "quality_quantiles":np.quantile(quality,[0,.1,.5,.9,1]).tolist(),
        "below_0_1":int((quality<.1).sum()),"simulation_validation_required":True}
    (assets/(args.name+"_quality.json")).write_text(json.dumps(report,indent=2))
    np.savez_compressed(assets/(args.name+"_tet.npz"),points=points,tets=tets)
    print(json.dumps(report))

if __name__=="__main__": main()
