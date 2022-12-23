import random
import os

import pyqrcodeng as pyqrcode #!not a pyqrcode
import numpy as np
import uuid
import cv2

class QRCodeGen:
    DEFAULT_ERROR_LEVELS = ['L', 'M', 'Q', 'H']
    DEFAULT_VERSIONS = list(range(1, 41))
    DEFAULT_MODES = 'binary', 'numeric', 'alphanumeric'

    def __init__(self, versions = None, error_levels = None, modes = None):
        self.versions = versions or self.DEFAULT_VERSIONS
        self.error_levels = error_levels or self.DEFAULT_ERROR_LEVELS
        self.modes = modes or self.DEFAULT_MODES

    def __call__(self):
        version = self.choise_version()
        mode = self.choise_mode()
        error_level = self.choise_error_level()

        return pyqrcode.create(
            self.generate_data(error_level, version, mode),
            error=error_level,
            version=version,
            mode=mode,
        )

    def choise_version(self):
        return random.choice(self.versions)

    def choise_error_level(self):
        return random.choice(self.error_levels)

    def choise_mode(self):
        return random.choice(self.modes)

    def generate_data(self, error_level: str, version: int, mode: str):
        mode_id = pyqrcode.tables.modes[mode]
        capacity = pyqrcode.tables.data_capacity[version][error_level][mode_id]
        prev_capacity = 0 if version == 1 else pyqrcode.tables.data_capacity[version-1][error_level][mode_id]

        data_size = np.random.randint(prev_capacity, capacity)
        match mode:
            case 'alphanumeric':
                return "".join(map(chr, np.random.choice(list(pyqrcode.tables.ALPHANUMERIC_CHARS), data_size)))
            case 'numeric':
                return "".join(map(str, np.random.choice(list(range(10)), data_size)))
            case 'binary':
                return np.random.bytes(data_size)
            case _:
                assert False, f"unexpected mode {mode}"

class QRCodeMarkup:
    DEFAULT_QUIET_ZONE = 2
    DEFAULT_MARKUP_SCALE = 3
    DEFAULT_QR_CODE_SCALE = 9

    def __init__(self, markup_scale = None, qr_code_scale = None, quiet_zone = None):
        self.markup_scale: int = markup_scale or self.DEFAULT_MARKUP_SCALE
        self.qr_code_scale: int = qr_code_scale or self.DEFAULT_QR_CODE_SCALE
        self.quiet_zone = quiet_zone or self.DEFAULT_QUIET_ZONE

        assert 1 <= self.markup_scale <= self.qr_code_scale
        assert (self.qr_code_scale - self.markup_scale) % 2 == 0

    def save(self, qr_code: pyqrcode.QRCode, dataset_dir: str):
        size = qr_code.symbol_size(self.qr_code_scale, self.quiet_zone)
        module_size = qr_code.symbol_size(1, 0)[0]
        module_offset = (self.qr_code_scale - self.markup_scale) // 2

        blank = np.zeros(shape=(self.qr_code_scale,size[0]),dtype=bool)

        row = blank.copy()
        row[module_offset:module_offset+self.markup_scale, self.quiet_zone * self.qr_code_scale:size[1]-self.quiet_zone * self.qr_code_scale] = True

        quiet_rows = np.vstack(list(blank for _ in range(self.quiet_zone)))

        qr_rows = np.vstack(list(row for _ in range(module_size)))

        rows = np.vstack((quiet_rows, qr_rows, quiet_rows))
        columns = rows.T
        centroids = np.logical_and(rows, columns)

        filename = f"{dataset_dir}/{str(uuid.uuid4())}"
        qr_code.png(f"{filename}.qr.png",scale=self.qr_code_scale,quiet_zone=self.quiet_zone) # RGB image

        for markup, name in [(rows, "rows"), (columns, "columns"), (centroids, "centroids")]:
            cv2.imwrite(f"{filename}.{name}.png", np.uint8(markup)*255) # grayscale image


class DatasetGen:
    def __init__(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        assert os.path.isdir(path)

        self.path = path
        self.set_markup_parameters()
        self.set_qr_code_parameters()

    def set_markup_parameters(self, quiet_zone = None, markup_scale = None, qr_code_scale = None):
        self.quiet_zone = quiet_zone or QRCodeMarkup.DEFAULT_QUIET_ZONE
        self.markup_scale = markup_scale or QRCodeMarkup.DEFAULT_MARKUP_SCALE
        self.qr_code_scale = qr_code_scale or QRCodeMarkup.DEFAULT_QR_CODE_SCALE
        return self

    def set_qr_code_parameters(self, error_levels = None, versions = None, modes = None):
        self.error_levels = error_levels or QRCodeGen.DEFAULT_ERROR_LEVELS
        self.versions = versions or QRCodeGen.DEFAULT_VERSIONS
        self.modes = modes or QRCodeGen.DEFAULT_MODES
        return self

    def generate(self, size, clear_old = True):
        assert os.path.isdir(self.path)
        if clear_old:
            for f in os.listdir(self.path):
                os.remove(os.path.join(self.path, f))

        generator = QRCodeGen(versions=self.versions, error_levels=self.error_levels, modes=self.modes)
        markup = QRCodeMarkup(markup_scale=self.markup_scale, qr_code_scale=self.qr_code_scale, quiet_zone=self.quiet_zone)

        for _ in range(size):
            code = generator()
            markup.save(code, self.path)
        pass

