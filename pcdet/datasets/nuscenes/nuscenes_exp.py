import nuscenes_utils
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

train_scenes = splits.mini_train
val_scenes = splits.mini_val
version = 'v1.0-mini'
with_cam = True
data_path = '/home/ave/Documents/UniTR/data/nuscenes/v1.0-mini'

nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
available_scenes = nuscenes_utils.get_available_scenes(nusc)
available_scene_names = [s['name'] for s in available_scenes]
train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))
train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=10, with_cam=with_cam
    )