import numpy as np
from skimage.transform import rotate
from scipy import ndimage

class RandomCrop3D(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image, center, angle, shift = sample['image'], sample['center'], sample['angle'], sample['shift']
        d, h, w = image.shape[-3:]
        #print('d, h, w', d, h, w)
        new_d, new_h, new_w = self.output_size
        #print('new_d, new_h, new_w', new_d, new_h, new_w)

        if center:
            depth = center[0]
            top = center[1]
            left = center[2]
        else:
            depth = np.random.randint(0, d - new_d + 1)
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)

        image = image[:, depth: depth + new_d,
                      top: top + new_h,
                      left: left + new_w]
        center = [depth, top, left]

        return {'image': image,
                'center': center,
                'angle': angle,
                'shift': shift}

class RandomRotate3D(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image, center, angle, shift = sample['image'], sample['center'], sample['angle'], sample['shift']

        if angle==None:
            angle = np.random.randint(-45, 46)

        new_image = []
        for i in range(image.shape[0]):
            new_image.append(rotate(image[i, ...], angle, mode='edge'))

        image = np.array(new_image)
        return {'image': image,
                'center': center,
                'angle':angle,
                'shift': shift}
    
class RandomTranslate3D(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image, center, angle, shift = sample['image'], sample['center'], sample['angle'], sample['shift']

        [b,x,y,z] = image.shape
        
        if shift:
            [x_shift,y_shift,z_shift] = shift
        else:
            x_shift = np.random.randint(-self.output_size[0], self.output_size[0])
            y_shift = np.random.randint(-self.output_size[1], self.output_size[1])
            z_shift = np.random.randint(-self.output_size[2], self.output_size[2])
            shift = [x_shift,y_shift,z_shift]
            
        image_translated = np.ones_like(image) * -1
        image_translated[:, max(x_shift,0):x+min(x_shift,0), max(y_shift,0):y+min(y_shift,0), max(z_shift,0):z+min(z_shift,0)] = image[:, -min(x_shift,0):x-max(x_shift,0), -min(y_shift,0):y-max(y_shift,0), -min(z_shift,0):z-max(z_shift,0)]

        '''
        x_low = x_shift
        x_high = x + x_shift
        y_low = y_shift
        y_high = y + y_shift
        z_low = z_shift
        z_high = z + z_shift

        xl = np.linspace(x_low, x_high - 1, x)
        yl = np.linspace(y_low, y_high - 1, y)
        zl = np.linspace(z_low, z_high - 1, z)
        
        new_image = []
        for i in range(b):
            xx, yy, zz = np.meshgrid(xl, yl, zl, indexing='ij')
            new_image.append(ndimage.map_coordinates(image[i,...], [xx, yy, zz], order=1))
        '''
        return {'image': image_translated,
                'center': center,
                'angle':angle,
                'shift': shift}

class RandomFlip3D(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, dummy_input=1):
        self.dummy_input=dummy_input

    def __call__(self, sample):
        image, center, angle, shift = sample['image'], sample['center'], sample['angle'], sample['shift']

        if center:
            depth = center[0]
            top = center[1]
            left = center[2]
        else:
            depth = np.random.randint(0, 2)
            top = np.random.randint(0, 2)
            left = np.random.randint(0, 2)

        if depth%2 == 0:
            image = np.flip(image, 1)
        if top%2 == 0:
            image = np.flip(image, 2)
        if left%2 == 0:
            image = np.flip(image, 3)

        image = image.copy()

        return {'image': image,
                'center': center,
                'angle': angle,
                'shift': shift}