from dataset_subclasses import PennAction, Olympic, DogRun, Human, Birds, Cats

# paths
path_to_root = '/Users/leonardbereska/myroot/'
path_to_harddrive = '/Volumes/Uni/'

# relative path
# path = path_to_harddrive + 'owny/'
# assert os.path.exists(path)

# construct dataset
# test = Jockel(path_to_dataset=path, res=600)
# extract_frames(path, from_dir='exported', to_dir='raw', video_format='mp4', img_format='jpg', frame_rate=2)
# test.process(from_dir='raw', to_dir='processed')

# PennAction
# path = path_to_harddrive + 'penn/'
# penn = PennAction(path_to_dataset=path, make_trainset=True, res=600, from_dir='frames', selected_actions=["tennis_serve", "tennis_forehand", "baseball_pitch","baseball_swing", "jumping_jacks", "golf_swing"])
# penn.process()

# Olympic Sports
# path = path_to_harddrive + 'olympic/'
# olymp = Olympic(path_to_dataset=path, make_trainset=True, res=600, from_dir='frames', selected_actions=["long_jump"])
# olymp.action_process(from_dir='frames', to_dir='processed')

# Dogs
# path = path_to_harddrive + 'dogs/'
# dogs = DogRun(path, 480)
# dogs.process()

# Human
# path = path_to_harddrive + 'human/human_all_copy/S1/'
# human = Human(path, res=600, selected_actions=['posing'], use_mask=True, frame_rate=10)
# human.process()

# Birds
# path = path_to_root + 'birds/birds/'
# birds = Birds(path, res=720, make_trainset=True, from_dir='images')
# birds.process()

# Cats
path = path_to_harddrive + 'cats/cats2/'
cats = Cats(path_to_dataset=path, res=600, make_trainset=True, from_dir='train')
cats.process()
