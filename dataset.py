import logging
# import re

import cv2
import lmdb
import six
from fastai.vision import *
from torchvision import transforms

from transforms import CVColorJitter, CVDeterioration, CVGeometry
from utils import CharsetMapper, onehot


class ImageDataset(Dataset):
    "`ImageDataset` read data from LMDB database."

    def __init__(self,
                 path: PathOrStr,   #LMDB数据库的路径。
                 is_training: bool = True,  #是否为训练模式
                 img_h: int = 32,
                 img_w: int = 100,
                 max_length: int = 70,
                 check_length: bool = True,
                 case_sensitive: bool = False,
                 # charset_path: str = 'data/charset_vn_with_space.txt',
                 charset_path: str = 'data/tibetan_charset.txt', #字符集文件的路径。该文件包含了数据集中可能出现的字
                 convert_mode: str = 'RGB', #图像转换模式
                 data_aug: bool = True,
                 deteriorate_ratio: float = 0.,
                 multiscales: bool = True,
                 one_hot_y: bool = True,
                 return_idx: bool = False,
                 return_raw: bool = False,
                 **kwargs):
        print(f"baizhen____________开始解析数据库")
        self.path, self.name = Path(path), Path(path).name
        print(f"baizhen____________数据库path={self.path}")
        print(f"baizhen____________数据库name={self.path}")
        assert self.path.is_dir() and self.path.exists(), f"{path} is not a valid directory."
        self.convert_mode, self.check_length = convert_mode, check_length
        self.img_h, self.img_w = img_h, img_w
        self.max_length, self.one_hot_y = max_length, one_hot_y
        self.return_idx, self.return_raw = return_idx, return_raw
        self.case_sensitive, self.is_training = case_sensitive, is_training
        self.data_aug, self.multiscales = data_aug, multiscales
        self.charset = CharsetMapper(charset_path, max_length=max_length + 1)
        self.character = self.charset.label_to_char.values()
        self.c = self.charset.num_classes

        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        assert self.env, f'Cannot open LMDB dataset from {path}.'
        print(f"baizhen____________成功打开数据库{self.path}")
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode())) #在一个事务中，获取 LMDB 数据库中的样本数量，并将其转换为整数，并赋值给 self.length 属性。

        if self.is_training and self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])  #如果是训练模式且需要进行数据增强，则创建一个由多个数据增强操作组成的变换序列，并将其赋值给 self.augment_tfs 属性。
        self.totensor = transforms.ToTensor()   #创建一个 ToTensor 变换对象，并将其赋值给 self.totensor 属性。

    def __len__(self):
        #返回数据集长度
        return self.length

    def _next_image(self, index):
        #返回下一个图像的索引
        next_index = random.randint(0, len(self) - 1)
        return self.get(next_index)

    def _check_image(self, x, pixels=6):
        #检查图像是否符合指定的像素大小要求。
        if x.size[0] <= pixels or x.size[1] <= pixels:
            return False
        else:
            return True

    def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT):
        #多尺度调整图像大小的方法。
        def _resize_ratio(img, ratio, fix_h=True):
            if ratio * self.img_w < self.img_h:
                if fix_h:
                    trg_h = self.img_h
                else:
                    trg_h = int(ratio * self.img_w)
                trg_w = self.img_w
            else:
                trg_h, trg_w = self.img_h, int(self.img_h / ratio)
            img = cv2.resize(img, (trg_w, trg_h))
            pad_h, pad_w = (self.img_h - trg_h) / 2, (self.img_w - trg_w) / 2
            top, bottom = math.ceil(pad_h), math.floor(pad_h)
            left, right = math.ceil(pad_w), math.floor(pad_w)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            return im
        if self.is_training:
            if random.random() < 0.5:
                base, maxh, maxw = self.img_h, self.img_h, self.img_w
                h, w = random.randint(base, maxh), random.randint(base, maxw)
                return _resize_ratio(img, h / w)
            else:
                return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio
        else:
            return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio

    def resize(self, img):
        # 调整图像大小的方法。
        if self.multiscales:
            return self.resize_multiscales(img, cv2.BORDER_REPLICATE)
        else:
            return cv2.resize(img, (self.img_w, self.img_h))

    def get(self, idx):
        # 根据索引获取图像和标签。
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
            #根据idx生成图片和标签的键值对
            try:
                #baizhen——120
                label = str(txn.get(label_key.encode()), 'utf-8').strip('').replace(' ','').replace('་','').replace('།','') # 从数据库中根据label_key的键取出信息 label
                print(f"baizhen _______________________________ label={label}")
                print(f'baizhen!!!!!!!!!!!!!!!!!!!!!!!!lebel == {label}')
                #baizhen


                if not set(label).issubset(self.character):
                    print(f'baizhen!!!!!!!!!!!!!!!!!!!!!!!!lebel == {label}  中有不属于数据集的内容详情如下')
                    for i in label :
                        if not set(i).issubset(self.character):
                            print(f'{i} not in data')
                    #baizhen
                    #这行代码是我注释的 好让更多的数据通过检测，不满足的数据在上一个baizhen————120已经处理过了
                    # return self._next_image(idx)  # 这行代码检查标签中的字符是否都属于字符集self.character。如果有字符不属于字符集，则调用self._next_image(idx)方法获取下一张图像和标签。
                    #baizhen

                # label = re.sub('[^0-9a-zA-Z]+', '', label)
                if self.check_length and self.max_length > 0:   #判断长度是否合法，不合法就跳过
                    if len(label) > self.max_length or len(label) <= 0:
                        # logging.info(f'Long or short text image is found: {self.name}, {idx}, {label}, {len(label)}')
                        print(f'baizhen!!!!!!!!!!!!!!!!!!!!!!!!lebel == {label}  长度不合法 长度为{len(label)}')
                        return self._next_image(idx)
                label = label[:self.max_length]
                print(f'baizhen!!!!!!!!!!!!!!!!!!!!!!!!lebel裁剪之后 == {label}')

                imgbuf = txn.get(image_key.encode())  # image
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
                    image = PIL.Image.open(buf).convert(self.convert_mode)
                if self.is_training and not self._check_image(image):
                    # logging.info(f'Invalid image is found: {self.name}, {idx}, {label}, {len(label)}')
                    return self._next_image(idx)
            except:
                print(f'baizhen!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!有一张图片处理失败 信息如下 name={self.name} idx={idx} label={label}')
                import traceback
                traceback.print_exc()
                logging.info(f'Corrupted image is found: {self.name}, {idx}, {label}, {len(label)}')
                return self._next_image(idx)
            return image, label, idx

    def _process_training(self, image):
        # 对训练图像进行处理的方法。
        if self.data_aug: image = self.augment_tfs(image)
        image = self.resize(np.array(image))
        return image

    def _process_test(self, image):
        # 对测试图像进行处理的方法。
        return self.resize(np.array(image))  # TODO:move is_training to here

    def __getitem__(self, idx):
        # 根据索引获取图像和标签的方法。
        image, text, idx_new = self.get(idx) #使用get方法获取图像，文本，新的索引
        # print(image, text, idx_new, idx)
        print(f'baizhen--------------------------------------------------------------------------------')
        print(f'baizhen----------idx{idx}       image.shape={image}       text{text}     idx_new{idx_new} ')
        # if not self.is_training: assert idx == idx_new, f'idx {idx} != idx_new {idx_new} during testing.' #在非训练模式下检查索引idx是否与新的索引值idx_new相等。如果不相等，会抛出一个断言错误。
        if not self.is_training:
            if idx != idx_new:
                print(f'baizhen----------idx{idx} 索引有误，重新定义了new——idx')
                idx_new = idx
        if self.is_training:
            image = self._process_training(image)
        else:
            image = self._process_test(image)
        if self.return_raw: return image, text
        image = self.totensor(image)

        length = tensor(len(text) + 1).to(dtype=torch.long)  # one for end token
        label = self.charset.get_labels(text, case_sensitive=self.case_sensitive)
        label = tensor(label).to(dtype=torch.long)
        if self.one_hot_y: label = onehot(label, self.charset.num_classes)

        if self.return_idx:
            y = [label, length, idx_new]
        else:
            y = [label, length]
        return image, y


class TextDataset(Dataset):
    def __init__(self,
                 path: PathOrStr,
                 delimiter: str = '\t',
                 max_length: int = 25,
                 charset_path: str = 'data/charset_vn.txt',
                 case_sensitive=False,
                 one_hot_x=True,
                 one_hot_y=True,
                 is_training=True,
                 smooth_label=False,
                 smooth_factor=0.2,
                 use_sm=False,
                 **kwargs):
        self.path = Path(path)
        self.case_sensitive, self.use_sm = case_sensitive, use_sm
        self.smooth_factor, self.smooth_label = smooth_factor, smooth_label
        self.charset = CharsetMapper(charset_path, max_length=max_length + 1)
        self.one_hot_x, self.one_hot_y, self.is_training = one_hot_x, one_hot_y, is_training
        if self.is_training and self.use_sm: self.sm = SpellingMutation(charset=self.charset)

        dtype = {'inp': str, 'gt': str}
        self.df = pd.read_csv(self.path, dtype=dtype, delimiter=delimiter, na_filter=False)
        self.inp_col, self.gt_col = 0, 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text_x = self.df.iloc[idx, self.inp_col].strip()
        # text_x = re.sub('[^0-9a-zA-Z]+', '', text_x)
        if not self.case_sensitive: text_x = text_x.lower()
        if self.is_training and self.use_sm: text_x = self.sm(text_x)

        length_x = tensor(len(text_x) + 1).to(dtype=torch.long)  # one for end token
        label_x = self.charset.get_labels(text_x, case_sensitive=self.case_sensitive)
        label_x = tensor(label_x)
        if self.one_hot_x:
            label_x = onehot(label_x, self.charset.num_classes)
            if self.is_training and self.smooth_label:
                label_x = torch.stack([self.prob_smooth_label(l) for l in label_x])
        x = [label_x, length_x]

        text_y = self.df.iloc[idx, self.gt_col]
        # text_y = re.sub('[^0-9a-zA-Z]+', '', text_y)
        if not self.case_sensitive: text_y = text_y.lower()
        length_y = tensor(len(text_y) + 1).to(dtype=torch.long)  # one for end token
        label_y = self.charset.get_labels(text_y, case_sensitive=self.case_sensitive)
        label_y = tensor(label_y)
        if self.one_hot_y: label_y = onehot(label_y, self.charset.num_classes)
        y = [label_y, length_y]

        return x, y

    def prob_smooth_label(self, one_hot):
        one_hot = one_hot.float()
        delta = torch.rand([]) * self.smooth_factor
        num_classes = len(one_hot)
        noise = torch.rand(num_classes)
        noise = noise / noise.sum() * delta
        one_hot = one_hot * (1 - delta) + noise
        return one_hot


class SpellingMutation(object):
    def __init__(self, pn0=0.7, pn1=0.85, pn2=0.95, pt0=0.7, pt1=0.85, charset=None):
        """ 
        Args:
            pn0: the prob of not modifying characters is (pn0)
            pn1: the prob of modifying one characters is (pn1 - pn0)
            pn2: the prob of modifying two characters is (pn2 - pn1), 
                 and three (1 - pn2)
            pt0: the prob of replacing operation is pt0.
            pt1: the prob of inserting operation is (pt1 - pt0),
                 and deleting operation is (1 - pt1)
        """
        super().__init__()
        self.pn0, self.pn1, self.pn2 = pn0, pn1, pn2
        self.pt0, self.pt1 = pt0, pt1
        self.charset = charset
        logging.info(f'the probs: pn0={self.pn0}, pn1={self.pn1} ' +
                     f'pn2={self.pn2}, pt0={self.pt0}, pt1={self.pt1}')

    def is_digit(self, text, ratio=0.5):
        length = max(len(text), 1)
        digit_num = sum([t in self.charset.digits for t in text])
        if digit_num / length < ratio: return False
        return True

    def is_unk_char(self, char):
        # return char == self.charset.unk_char
        return (char not in self.charset.digits) and (char not in self.charset.alphabets)

    def get_num_to_modify(self, length):
        prob = random.random()
        if prob < self.pn0:
            num_to_modify = 0
        elif prob < self.pn1:
            num_to_modify = 1
        elif prob < self.pn2:
            num_to_modify = 2
        else:
            num_to_modify = 3

        if length <= 1:
            num_to_modify = 0
        elif length >= 2 and length <= 4:
            num_to_modify = min(num_to_modify, 1)
        else:
            num_to_modify = min(num_to_modify, length // 2)  # smaller than length // 2
        return num_to_modify

    def __call__(self, text, debug=False):
        if self.is_digit(text): return text
        length = len(text)
        num_to_modify = self.get_num_to_modify(length)
        if num_to_modify <= 0: return text

        chars = []
        index = np.arange(0, length)
        random.shuffle(index)
        index = index[: num_to_modify]
        if debug: self.index = index
        for i, t in enumerate(text):
            if i not in index:
                chars.append(t)
            elif self.is_unk_char(t):
                chars.append(t)
            else:
                prob = random.random()
                if prob < self.pt0:  # replace
                    chars.append(random.choice(self.charset.alphabets))
                elif prob < self.pt1:  # insert
                    chars.append(random.choice(self.charset.alphabets))
                    chars.append(t)
                else:  # delete
                    continue
        new_text = ''.join(chars[: self.charset.max_length - 1])
        return new_text if len(new_text) >= 1 else text
