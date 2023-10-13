"""
NOTES: Dataset details summary
- Image dataset, regression, 9 silos, 11k images in total
- Filename interpretation:
  For
    ADNI_016_S_0702_PT_AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_Br_20100927170934088_44_S92690_I193766.png
  the patient id is 0702, and image id is I193766.
- A silo is a manufacturer of a machine that took the images
- Each patient can have multiple images
- Each image is of size 160 x 160 by default (but scaled down below)
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image


class ADNI(torch.utils.data.Dataset):

  def __init__(self, examples, transform=None, target_transform=None):
    self.examples = examples

    self.transform = transform
    self.target_transform = target_transform

  def read_image(path: str):
    with Image.open(path) as im:
      img = np.array(im.resize((32, 32)), dtype=np.float32)
    return img / 255.0
  def read_data(seed=ROOT_SEED, test_split=0.25, save_dir=None):
    print(f'(seed={seed}) Reading ADNI data from {DATA_PATH} and {LABEL_PATH}...')
    # A nested map dic[patient_id][image_id] = label
    uid_mid_to_label = {}

    with open(LABEL_PATH, 'r') as f:
      labels = f.readlines()
    for label in labels:
      tokens = label.strip().split(',')
      uid, mid, value = tokens
      if uid in uid_mid_to_label:
        uid_mid_to_label[uid][mid] = float(value)
      else:
        uid_mid_to_label[uid] = {mid: float(value)}

    print('Populated label mapping...')
    # Partitioned by manufacturers (silos)
    x_trains, y_trains, x_tests, y_tests = [], [], [], []

    client_id = 0
    for client in os.listdir(DATA_PATH):
      if client not in MANUFACTURERS:
        continue

      print(f'Processing client "{client}"...')
      # Images and labels for each silo
      xx, yy = [], []
      images = os.listdir(os.path.join(DATA_PATH, client))
      for img_path in images:
        if not img_path.endswith('png'):  # ignore DS_Store
          continue
        uid, mid = img_path.strip().split('_')[3], img_path[:-4].strip().split('_')[-1]
        if uid in uid_mid_to_label:
          xx.append(read_image(os.path.join(DATA_PATH, client, img_path)))
          yy.append(uid_mid_to_label[uid][mid])

      # Skip if no samples for a silo
      assert len(xx) == len(yy)
      if len(xx) == 0:
        print(f'No data from client "{client}"')
        continue

      features, labels = np.array(xx)[..., None], np.array(yy)
      print('(unsplit) features:', features.shape, 'labels:', labels.shape)

      # Seeds controls the shuffling + train/test split
      client_seed = seed + client_id
      x_train, x_test, y_train, y_test = train_test_split(
          features, labels, test_size=test_split, random_state=client_seed)

      x_trains.append(x_train)
      y_trains.append(y_train)
      x_tests.append(x_test)
      y_tests.append(y_test)
      client_id += 1

    # List[List[array]], List[List[float]]
    x_trains = np.array(x_trains, dtype=object)
    y_trains = np.array(y_trains, dtype=object)
    x_tests = np.array(x_tests, dtype=object)
    y_tests = np.array(y_tests, dtype=object)

    if save_dir is not None:
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      np.save(os.path.join(save_dir, f'train_images_seed{seed}'), x_trains)
      np.save(os.path.join(save_dir, f'train_labels_seed{seed}'), y_trains)
      np.save(os.path.join(save_dir, f'test_images_seed{seed}'), x_tests)
      np.save(os.path.join(save_dir, f'test_labels_seed{seed}'), y_tests)
      print(f'Saved preprocessed ADNI dataset to {save_dir}')

    return x_trains, y_trains, x_tests, y_tests


class ADNIData():
  """
  adni allocator.
  """

  def __init__(self, client_num, sample_rate=-1, data_sharing=False):
    MANUFACTURERS = {
      'CPS',
      'GE_MEDICAL_SYSTEMS',
      'GEMS',
      'HERMES',
      'MiE',
      'Multiple',
      'Philips',
      'Philips_Medical_Systems',
      'SIEMENS',
      'Siemens_ECAT',
      'Siemens_CTI'
    }

    DATA_DIR = './dataset/adni/'
    DATA_PATH = DATA_DIR + 'adni_data2/'
    LABEL_PATH = DATA_DIR + 'labels_2.txt'
    ROOT_SEED = int((np.e ** np.euler_gamma) ** np.pi * 1000)

    self.root = os.path.expanduser(DATA_DIR)

    if not self._check_exists():
      print(self.root)
      raise RuntimeError('Dataset not found.')

    self.train_dict, total_train = self._read_file(self.training_file)
    self.test_dict, total_test = self._read_file(self.test_file)

    self.data_sharing = data_sharing
    if not 0 < sample_rate <= 1:
      sample_rate = 1
    np.random.shuffle(total_train)
    np.random.shuffle(total_test)
    self.share_train = total_train[:int(sample_rate * len(total_train))]
    self.share_test = total_test[:int(sample_rate * len(total_test))]

    self.num_local_train = len(total_train) // client_num
    self.num_local_test = len(total_test) // client_num

    normalize = transforms.Normalize(mean=[0.131], std=[0.308])
    self.train_transform = transforms.Compose([
      transforms.RandomResizedCrop(28),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize
    ])
    self.test_transform = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])

  def create_dataset_for_center(self, batch_size, num_workers):
    _train_set = ADNI(self.share_train, self.train_transform)
    _test_set = ADNI(self.share_test, self.test_transform)
    train_loader = DataLoader(_train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(_test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, len(_train_set)

  def create_dataset_for_client(self, distribution, batch_size, num_workers, subset=tuple(range(10))):
    """
    subset: construct local data set with certain label(s).
    distribution: the distribution (of label space) to construct local data set.
    """
    distribution = np.asarray(distribution) / np.sum(distribution)

    def sample_data(data_dict, local_num):
      local_data = list()
      for i, p in enumerate(distribution):
        snum = int(local_num * p)
        indices = np.random.choice(len(data_dict[i]), snum, replace=False)
        local_data.extend([(k, i) for k in data_dict[i][indices]])
      return local_data

    local_train, local_test = list(), list()
    if len(subset) < 10:
      for i in subset:
        local_train.extend([(k, i) for k in self.train_dict[i]])
        local_test.extend([(k, i) for k in self.test_dict[i]])
    else:
      local_train = sample_data(self.train_dict, self.num_local_train)
      local_test = sample_data(self.test_dict, self.num_local_test)

    if self.data_sharing:
      local_train.extend(self.share_train)
      local_test.extend(self.share_test)

    _train_set = ADNI(local_train, self.train_transform)
    _test_set = ADNI(local_test, self.test_transform)
    train_loader = DataLoader(_train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(_test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, len(local_train)

  @property
  def processed_folder(self):
    return os.path.join(self.root, 'processed')

  def _check_exists(self):
    return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
           os.path.exists(os.path.join(self.processed_folder, self.test_file))

  def _read_file(self, data_file):
    """
    return:
        data: (dict: {label: array[images, ...]} )
        total_data: (list [(image, label), ...] )
    """
    data = {i: [] for i in range(10)}
    total_data, total_targets = torch.load(os.path.join(self.processed_folder, data_file))
    total_data = [x.numpy() for x in total_data]
    for k, v in zip(total_data, total_targets):
      data[int(v)].append(k)
    for k, v in data.items():
      data[k] = np.asarray(v)
    return data, list(zip(total_data, total_targets))


if __name__ == "__main__":
  pass
