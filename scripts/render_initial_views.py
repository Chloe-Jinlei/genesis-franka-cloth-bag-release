#!/usr/bin/env python3
"""Render the recorded time-zero state from four previous dressing cameras.

No integration or soft-state setter is used. Building the original meshes and
saved initial robot q must reproduce the input state before rendering is allowed.
The process-local camera factory is replaced only to add the four cameras before
Genesis Scene.build(); no source file or physical entity is edited.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

REPO = Path(__file__).resolve().parents[1]
CAMERAS = (
    ('initial_01_perspective',(1.65,-2.1,1.85),(0.,0.,1.),42.),
    ('initial_02_top',(0.,-.001,2.85),(0.,0.,.72),35.),
    ('initial_03_side',(-2.,-.3,1.50),(0.,0.,1.02),42.),
    ('initial_04_back',(0.,2.4,1.65),(0.,0.,1.02),42.),
)
REFERENCE = {
    'repository':'https://github.com/Chloe-Jinlei/four-view-genesis',
    'revision':'049bc52b471baf13e6a949e5ad5015d115927560',
    'camera_source':'simulate.py',
    'camera_source_url':'https://github.com/Chloe-Jinlei/four-view-genesis/blob/049bc52b471baf13e6a949e5ad5015d115927560/simulate.py#L119-L128',
    'view_name_source':'README.md: Four original input images; oblique/front perspective, top, side, back',
    'previous_native_resolution_px':[960,600],
    'scope':'Exact camera poses and FOV from the previous public dressing reconstruction, not recovered calibration of its original reference photographs.',
    'extra_detail_camera_excluded':True,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene-dir',type=Path,default=REPO/'teddy_bag_flat_start')
    ap.add_argument('--state',type=Path,default=REPO/'media/initial_views/source_initial_state.npz')
    ap.add_argument('--config',type=Path,default=REPO/'media/initial_views/source_config.json')
    ap.add_argument('--output',type=Path,default=REPO/'media/initial_views')
    ap.add_argument('--width',type=int,default=1600)
    ap.add_argument('--height',type=int,default=1000)
    args=ap.parse_args()
    if args.width<=0 or args.height<=0 or args.width*600!=args.height*960:
        raise ValueError('Keep the original 8:5 camera aspect ratio')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    paths=[out/(name+'.png') for name,*_ in CAMERAS]+[out/'initial_views.json']
    if any(p.exists() for p in paths):raise FileExistsError('Refusing to replace previously rendered evidence; use a new --output')
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    sys.path.insert(0,str(args.scene_dir.resolve()))
    import numpy as np
    from PIL import Image
    import scene as simulation
    import initial_views
    import genesis as gs
    with np.load(args.state,allow_pickle=False) as z:
        saved={k:np.array(z[k],copy=True) for k in z.files}
    if float(saved['time'])!=0.:raise ValueError('Only a recorded time-zero state is accepted')
    cfg=json.loads(args.config.read_text())
    def add_cameras(scene):
        return {name:scene.add_camera(model='pinhole',res=(args.width,args.height),pos=pos,
            lookat=lookat,up=(0.,0.,1.),fov=fov,GUI=False) for name,pos,lookat,fov in CAMERAS}
    def no_step(*a,**kw):raise RuntimeError('Physics stepping is forbidden during initial-view export')
    initial_views.add_initial_cameras=add_cameras
    original_step=gs.Scene.step;gs.Scene.step=no_step
    try:
        scene,arms,bag,bear,cameras=simulation.build(cfg,viewer=False,width=320,height=200,
            initial_q={s:saved['q_'+s].reshape(-1) for s in ('left','right')})
        def snapshot():
            return {'bag':simulation.positions(bag).copy(),'bear':simulation.positions(bear).copy(),
                **{'q_'+s:r.get_qpos().detach().cpu().numpy().reshape(-1).copy() for s,r in arms.items()}}
        before=snapshot()
        errors={k:float(np.max(np.abs(v-saved[k].reshape(v.shape)))) for k,v in before.items()}
        if max(errors.values())>1e-12:raise RuntimeError(f'Rebuilt initial state differs from recorded state: {errors}')
        def clock():
            return {'genesis_time_s':float(scene.sim.cur_t),
                'genesis_steps':np.asarray(scene.sim.steps.detach().cpu().numpy()).reshape(-1).tolist(),
                'native_ipc_frame':int(scene.sim.coupler._ipc_world.frame())}
        start=clock()
        if start['genesis_time_s']!=0. or any(start['genesis_steps']) or start['native_ipc_frame']!=0:
            raise RuntimeError('Scene has already advanced')
        images=[]
        for name,pos,lookat,fov in CAMERAS:
            rgb=np.asarray(cameras[name].render(rgb=True,depth=False,segmentation=False,normal=False,
                antialiasing=False,force_render=True)[0])
            if rgb.shape!=(args.height,args.width,3) or rgb.dtype!=np.uint8:raise RuntimeError('Unexpected native RGB format')
            dest=out/(name+'.png');Image.fromarray(rgb).save(dest)
            images.append({'file':dest.name,'sha256':sha(dest),'dimensions_px':[args.width,args.height],
                'camera':{'model':'pinhole','position_m':list(pos),'lookat_m':list(lookat),'up':[0.,0.,1.],'fov_degrees':fov}})
        after=snapshot();end=clock()
        if start!=end or any(not np.array_equal(before[k],after[k]) for k in before):
            raise RuntimeError('Physical state changed during camera-only rendering')
        report={'status':'rendered_verified_recorded_initial_state','state_time_s':0.,
            'source_run':'flat_full_17','source_state_sha256':sha(args.state),'source_config_sha256':sha(args.config),
            'reference':REFERENCE,'renderer':'Genesis native camera, original scene geometry and surfaces',
            'camera_resolution_policy':'Original 8:5 aspect ratio; native resolution increased from 960x600 to 1600x1000 by default',
            'physics_steps_executed':0,'soft_state_setters_used':False,'physical_state_unchanged':True,
            'initial_state_max_absolute_errors':errors,'clock_before':start,'clock_after':end,
            'image_processing':'Native RGB encoded losslessly as PNG; no pixel edits','images':images}
        (out/'initial_views.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'images':[v['file'] for v in images],'errors':errors}),flush=True)
    finally:
        gs.Scene.step=original_step


if __name__=='__main__':main()
