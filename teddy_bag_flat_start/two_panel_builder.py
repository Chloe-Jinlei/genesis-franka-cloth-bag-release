"""Build a soft tote from exactly two fabric panels sewn along a U seam.

There is no side panel, bottom panel, gusset, hidden support or rigid skeleton.
The shared bottom seam has zero transverse depth. A gently opened rest shape
keeps the two finite-thickness fabric layers apart for IPC initialization.
All previous prototype assets remain untouched.
"""
from __future__ import annotations
import json
from numbers import Real
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import trimesh
from assets_builder import ASSETS, topology, write_obj

W, H, A = .42, .34, .07


def opened_panel_geometry():
    """Preserve warp/weft arc lengths in a smooth, mildly sheared opening.

Projection widths and column Z scales are alternately fitted. This is an
initial geometric approximation to unfolded rectangular cloth, not a claim of
zero strain/shear or an experimentally measured drape. Discrete metrics are
reported separately for each resolution.
"""
    us, ts = np.linspace(0,1,301), np.linspace(0,1,301)
    f = np.sqrt(np.maximum(np.sin(np.pi*us),0))
    g = np.sqrt(np.maximum(np.sin(np.pi*ts/2),0))
    f[[0,-1]], g[0] = 0., 0.
    widths, heights = np.full(len(ts),W), np.full(len(us),H)
    for _ in range(7):
        for j,t in enumerate(ts):
            dy = np.diff(A*f*g[j]); dz = np.diff(heights*t)
            widths[j] = brentq(lambda w: np.sum(np.sqrt((w*np.diff(us))**2+dy*dy+dz*dz))-W,.01,W)
        for i,u in enumerate(us):
            dx = np.diff(widths*(u-.5)); dy = np.diff(A*f[i]*g)
            heights[i] = brentq(lambda h: np.sum(np.sqrt(dx*dx+dy*dy+(h*np.diff(ts))**2))-H,.01,H)
    def point(u,t,side):
        width = np.interp(t,ts,widths)
        height = np.interp(u,us,heights)
        yy = A*np.sqrt(max(np.sin(np.pi*u),0))*np.sqrt(max(np.sin(np.pi*t/2),0))
        if u <= 1e-10 or u >= 1-1e-10 or t <= 1e-10:
            yy = 0.
        return [width*(u-.5),side*yy,height*t]
    return point,{"target_flat_width_m":W,"target_flat_height_m":H,
                  "top_projected_width_m":float(widths[-1]),
                  "top_center_gap_m":2*A,
                  "center_top_height_m":float(heights[len(us)//2]),
                  "side_seam_top_height_m":float(heights[0]),
                  "method":"Alternating row/column arc-length fit of an initially opened two-panel surface; sqrt-sine separation prevents near-seam initial cloth overlap."}


def build_bag(spacing,stem,point,geometry,*,handle_extra=.04):
    """Build the same two-panel body with optional straight handle-leg length."""
    if isinstance(handle_extra,(bool,np.bool_)) or not isinstance(handle_extra,Real) \
            or not np.isfinite(handle_extra) or handle_extra<=0:
        raise ValueError("handle_extra must be a finite positive length in metres")
    handle_extra=float(handle_extra)
    across = np.unique(np.round(np.r_[np.linspace(-.0125,.0125,6),[-.01,0.,.01]],10))
    material_x = np.unique(np.round(np.r_[np.linspace(-W/2,W/2,round(W/spacing)+1),
                                         -.12+across,.12+across],10))
    us = material_x/W+.5
    ts = np.linspace(0,1,round(H/spacing)+1)
    points,faces,index = [],[],{}
    def vertex(p):
        key=tuple(np.round(p,9))
        if key not in index:
            index[key]=len(points);points.append(p)
        return index[key]
    def quad(a,b,c,d,flip=False):
        ff=[[a,b,d],[a,d,c]]
        faces.extend([f[::-1] for f in ff] if flip else ff)
    grids,face_groups={},{}
    groups={}
    for name,side in (("front",-1),("back",1)):
        grid=np.array([[vertex(point(u,t,side)) for u in us] for t in ts])
        begin=len(faces)
        for j in range(len(ts)-1):
            for i in range(len(us)-1):
                a,b,c,d=grid[j,i],grid[j,i+1],grid[j+1,i],grid[j+1,i+1]
                if j==0 and i==len(us)-2:
                    # At the bottom-right U-seam corner, the other diagonal
                    # would create a triangle made of three shared seam nodes,
                    # duplicated by the second panel. Both triangles must
                    # include the one interior node of this corner cell.
                    ff=[[a,b,c],[b,d,c]]
                    faces.extend([f[::-1] for f in ff] if side==1 else ff)
                else:
                    quad(a,b,c,d,flip=side==1)
        grids[name]=grid
        face_groups["panel_"+name+"_face_indices"]=np.arange(begin,len(faces),dtype=np.int32)
        groups["panel_"+name]=np.unique(grid).tolist()
    body_face_indices=np.arange(len(faces),dtype=np.int32)
    body_indices=np.arange(len(points),dtype=np.int32)
    groups["body"]=body_indices.tolist()
    groups["rim"]=np.unique(np.r_[grids['front'][-1],grids['back'][-1]]).tolist()
    groups["bottom_seam"]=grids["front"][0].tolist()
    groups["seam_left"]=grids["front"][:,0].tolist()
    groups["seam_right"]=grids["front"][:,-1].tolist()
    groups["u_seam"]=np.unique(np.r_[grids['front'][0],grids['front'][:,0],grids['front'][:,-1]]).tolist()
    assert all(np.array_equal(grids['front'][sl],grids['back'][sl]) for sl in
               ((0,slice(None)),(slice(None),0),(slice(None),-1)))
    # Handle cross-sections at each sewn root use the actual curved top row.
    # This preserves topology and avoids a sliver or a disconnected visual band.
    handle_data={}
    for name,side in (("front",-1),("back",1)):
        root_ids=[]
        for center in (-.12,.12):
            chosen=np.flatnonzero(np.abs(material_x-center)<=.0125+1e-9)
            root_ids.append(grids[name][-1,chosen])
        assert len(root_ids[0])==len(root_ids[1])
        root0=np.asarray(points)[root_ids[0]]
        root1=np.asarray(points)[root_ids[1]][::-1]
        center0=np.asarray(point(-.12/W+.5,1,side))
        center1=np.asarray(point(.12/W+.5,1,side))
        r=(center1[0]-center0[0])/2
        extra=handle_extra
        intervals=max(4,2*int(np.ceil(np.pi*r/spacing/2)))
        arc=np.linspace(np.pi,0,intervals+1)
        nz=max(2,int(np.ceil(extra/spacing)))
        xz=np.r_[np.column_stack([np.full(nz,-r),np.linspace(0,extra,nz+1)[:-1]]),
                 np.column_stack([r*np.cos(arc),extra+r*np.sin(arc)]),
                 np.column_stack([np.full(nz,r),np.linspace(extra,0,nz+1)[1:]])]
        xz[:,1]+=center0[2]
        frac=(xz[:,0]+r)/(2*r)
        yy=abs(center0[1])+(.085-abs(center0[1]))*np.sin(np.pi*frac)
        path=np.column_stack([xz[:,0],side*yy,xz[:,1]])
        tangent=np.gradient(xz,axis=0);tangent[0]=[0,1];tangent[-1]=[0,-1]
        tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        normals=np.column_stack([tangent[:,1],np.zeros(len(path)),-tangent[:,0]])
        arclength=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        root_offsets0=root0-center0;root_offsets1=root1-center1
        cross_s=material_x[np.abs(material_x+.12)<=.0125+1e-9]+.12
        strip=[]
        for j,p in enumerate(path):
            w0=max(1-arclength[j]/.04,0);w1=max(1-(arclength[-1]-arclength[j])/.04,0)
            offsets=(1-w0-w1)*cross_s[:,None]*normals[j]+w0*root_offsets0+w1*root_offsets1
            strip.append([vertex(v) for v in p+offsets])
        strip=np.asarray(strip)
        assert np.array_equal(strip[0],root_ids[0]) and np.array_equal(strip[-1],root_ids[1][::-1])
        for j in range(len(path)-1):
            for i in range(len(cross_s)-1):
                quad(strip[j,i],strip[j,i+1],strip[j+1,i],strip[j+1,i+1],flip=side==1)
        groups['handle_'+name]=np.unique(strip).tolist()
        groups['handle_'+name+'_root_left']=root_ids[0].tolist()
        groups['handle_'+name+'_root_right']=root_ids[1].tolist()
        # Alternating X-offset grasps keep the two Panda palms clear. These are
        # labels of real cloth vertices, never fixed or scripted particles.
        for label,xc in (("center",0.),("x_minus",-.05),("x_plus",.05)):
            rows=np.abs(path[:,0]-xc)<=max(.008,spacing*.55)
            if not rows.any():rows[np.argmin(np.abs(path[:,0]-xc))]=True
            groups['grasp_'+name+'_'+label]=np.unique(strip[rows]).tolist()
        groups['grasp_'+name]=groups['grasp_'+name+('_x_minus' if name=='front' else '_x_plus')]
        handle_data[name]={"root_center_left_m":center0.tolist(),"root_center_right_m":center1.tolist(),
                           "centerline_length_m":float(arclength[-1]),"cross_samples":len(cross_s),
                           "vertical_leg_extra_m":handle_extra,"semicircle_radius_m":float(r)}
    points,faces=np.asarray(points,float),np.asarray(faces,np.int32)
    mesh=trimesh.Trimesh(points,faces,process=False);trimesh.repair.fix_normals(mesh);faces=mesh.faces.copy()
    stats=topology(points,faces);body_stats=topology(points[body_indices],faces[body_face_indices])
    assert stats['connected_components']==1 and stats['boundary_loops']==3
    assert stats['nonmanifold_edges']==0 and stats['boundary_degree_two'] and stats['consistent_winding']
    assert body_stats['boundary_loops']==1 and not body_stats['watertight']
    assert stats['degenerate_triangles']==0
    assert np.ptp(points[groups['bottom_seam'],1])==0 and np.ptp(points[groups['bottom_seam'],2])==0
    panel_metrics={}
    for name in ('front','back'):
        grid=points[grids[name]]
        row_lengths=np.linalg.norm(np.diff(grid,axis=1),axis=-1).sum(axis=1)
        col_lengths=np.linalg.norm(np.diff(grid,axis=0),axis=-1).sum(axis=0)
        panel_metrics[name]={"row_lengths_m":row_lengths.tolist(),"column_lengths_m":col_lengths.tolist(),
                             "row_length_range_m":[float(row_lengths.min()),float(row_lengths.max())],
                             "column_length_range_m":[float(col_lengths.min()),float(col_lengths.max())],
                             "surface_area_m2":float(mesh.area_faces[face_groups['panel_'+name+'_face_indices']].sum()),
                             "nominal_flat_panel_area_m2":W*H}
    sep=np.linalg.norm(points[grids['front'][1:-1,1:-1]]-points[grids['back'][1:-1,1:-1]],axis=-1)
    min_sep=float(sep.min())
    assert min_sep>.0015
    uv=np.column_stack([(points[:,0]+W/2)/W,points[:,2]/.5])
    write_obj(ASSETS/(stem+'.obj'),points,faces,uv,'cotton_canvas.png')
    write_obj(ASSETS/(stem+'_collision.obj'),points,faces)
    np.savez_compressed(ASSETS/(stem+'.npz'),points=points,faces=faces,uv=uv,
                        body_face_indices=body_face_indices,
                        panel_front_grid=grids['front'],panel_back_grid=grids['back'],
                        material_x=material_x,material_height=ts*H,
                        **face_groups,**{k:np.asarray(v,np.int32) for k,v in groups.items()})
    centers={k:points[v].mean(0).tolist() for k,v in groups.items() if k.startswith('grasp')}
    return {"file":stem+'.obj',"npz_file":stem+'.npz',"collision_file":stem+'_collision.obj',
            "construction":"Exactly two whole cloth panels share the left, bottom and right U-shaped seam. There is no bottom panel, side panel or gusset.",
            "width_m":W,"height_m":H,"depth_m":0.,"top_opening_center_gap_m":2*A,
            "strap_width_m":.025,"strap_peak_z_m":float(points[:,2].max()),
            "bottom_seam_transverse_depth_m":0.,"bottom_seam_has_area":False,
            "geometry":geometry,"mesh":stats,"body_mesh":body_stats,
            "vertex_groups":groups,"body_face_indices":body_face_indices.tolist(),
            "panel_metrics":panel_metrics,"handles":handle_data,
            "grasp_centers_local_m":centers,"default_grasp_note":"Front handle uses its real x-minus patch; back uses x-plus. This offsets the two palms to avoid collision.",
            "body_sampling_m":spacing,"handle_cross_sampling_max_m":.005,
            "min_corresponding_interior_panel_gap_m":min_sep,
            "initial_separation_note":"Matched interior vertices exceed 1.5mm separation; this is a geometric screen, not a full IPC triangle-triangle collision clearance proof. Shared U-seam vertices are intentionally identical.",
            "suggested_material":{"name":"light cotton canvas","thickness_m":.0006,
                                  "surface_density_kg_m2":.24,"young_modulus_Pa":1.5e7,"poisson_ratio":.3,"contact_friction":.55},
            "estimated_mass_kg":float(mesh.area*.24),"fixed_particles":0,"hidden_rigid_support":False}


def render_preview(fine):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    data=np.load(ASSETS/fine['npz_file']);p,f=data['points'],data['faces'];tri=p[f]
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.linalg.norm(n,axis=1)[:,None]
    light=np.array([-.4,-.7,1.]);light/=np.linalg.norm(light)
    colors=np.array([.9,.85,.73])[None,:]*(.55+.45*np.abs(n@light))[:,None]
    fig=plt.figure(figsize=(13,6.5),facecolor='#eeeeec')
    for i,(title,angle) in enumerate((('Two fabric panels · open mouth',-58),('Side view · one bottom seam',0))):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        ax.add_collection3d(Poly3DCollection(tri,facecolors=colors,edgecolors='none',zsort='average'))
        seam=p[data['bottom_seam']];ax.plot(seam[:,0],seam[:,1],seam[:,2],color='#665743',linewidth=2)
        lo=p.min(0);hi=p.max(0);ax.set_xlim(lo[0]-.015,hi[0]+.015);ax.set_ylim(-.13,.13);ax.set_zlim(-.015,hi[2]+.01)
        ax.set_box_aspect([hi[0]-lo[0]+.03,.26,hi[2]+.025]);ax.view_init(elev=18 if i==0 else 9,azim=angle)
        ax.set_axis_off();ax.set_title(title,fontsize=14)
    fig.tight_layout();fig.savefig(ASSETS/'two_panel_preview.png',dpi=160,bbox_inches='tight');plt.close(fig)


def main():
    point,geometry=opened_panel_geometry()
    fine=build_bag(.01,'two_panel_bag',point,geometry)
    coarse=build_bag(.02,'two_panel_bag_coarse',point,geometry)
    # Same actual mass across diagnostic resolutions, despite tiny area changes.
    scale=fine['mesh']['triangle_area_m2']/coarse['mesh']['triangle_area_m2']
    coarse['estimated_mass_kg']=fine['estimated_mass_kg']
    coarse['suggested_material']['surface_density_kg_m2']*=scale
    coarse['production_rho_multiplier_for_equal_mass']=scale
    bear=json.loads((ASSETS/'metadata.json').read_text())['bear']
    common={"units":"metres","generator":"two_panel_builder.py","bear":bear}
    (ASSETS/'metadata_two_panel.json').write_text(json.dumps({**common,"bag":fine,"coarse_metadata_file":"metadata_two_panel_coarse.json"},indent=2)+'\n')
    (ASSETS/'metadata_two_panel_coarse.json').write_text(json.dumps({**common,"bag":coarse,"fine_metadata_file":"metadata_two_panel.json"},indent=2)+'\n')
    render_preview(fine)
    print(json.dumps({"fine":{k:fine[k] for k in ('file','mesh','geometry','grasp_centers_local_m','min_corresponding_interior_panel_gap_m')},
                      "coarse":{k:coarse[k] for k in ('file','mesh','grasp_centers_local_m','min_corresponding_interior_panel_gap_m')},
                      "preview":str(ASSETS/'two_panel_preview.png')},indent=2))

if __name__=='__main__':main()
