from struct import unpack
from utils import *

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}

class JPEG:
    def __init__(self, image_file):
        self.sampling = {}
        self.huffman_tables = {}
        self.quant = {}
        self.quantMapping = {}
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            else:
                lenchunk, = unpack(">H", data[2:4])
                lenchunk += 2
                data_chunk = data[4:lenchunk]
                if marker == 0xffe0:
                    pass
                elif marker == 0xffdb:
                    QT_id, QT = get_DQT(data_chunk)
                    self.quant[QT_id] = QT
                elif marker == 0xffc0:
                    get_info(data_chunk, self)
                elif marker == 0xffc4:
                    get_CHT(data_chunk, self)
                elif marker == 0xffda:
                    scan_data = data[lenchunk:]
                    scan_data, scan_data_len = RemoveFF00(scan_data)
                    scan_data = BitStream(scan_data)
                    self.block_data_Y, self.block_data_Cb, self.block_data_Cr = get_scan_data(scan_data, self)
                    lenchunk = lenchunk + scan_data_len
                data = data[lenchunk:]            
            if len(data)==0:
                break

if __name__ == "__main__":
    # img = JPEG('profile.jpg')
    img = JPEG('lena.jpg')
    img.decode()
    DrawImage(img)