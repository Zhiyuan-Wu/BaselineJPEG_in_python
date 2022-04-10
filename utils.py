from struct import unpack
import numpy as np
import scipy.fft
from PIL import Image

def get_bytes(data, n, offset=0):
    return unpack("B"*n, data[offset: offset+n])

def get_DQT(data):
    data_len = len(data)
    if data_len != 65:
        raise Exception("DQT data block error.")
    offset = 0
    qt_id = get_bytes(data, 1, offset)[0]
    offset += 1
    qt = get_bytes(data, 64, offset)
    
    return qt_id, qt

def get_info(data, obj):
    data_len = len(data)
    precision, obj.height, obj.width, components = unpack(">BHHB",data[0:6])
    for i in range(components):
        id, HV, QtbId = unpack("BBB",data[6+i*3:9+i*3])
        obj.sampling[id] = (HV>>4, HV&0x0F)
        obj.quantMapping[id] = QtbId
    obj.MaxH = max([H for H,V in obj.sampling.values()])
    obj.MaxV = max([V for H,V in obj.sampling.values()])

class BitStream():
    def __init__(self, data) -> None:
        self.data = data
        self.pos = 0

    def GetBit(self):
        b = self.data[self.pos >> 3]
        s = 7-(self.pos & 0x7)
        self.pos+=1
        return (b >> s) & 1

    def GetBitN(self, n):
        val = 0
        for i in range(n):
            val = val*2 + self.GetBit()
        return val

class CHT:
    def __init__(self, num_symbol, symbol_list):
        self.num_symbol = num_symbol
        self.symbol_list = symbol_list
        self.first = [0] * 16
        for i in range(1, 16):
            self.first[i] = 2 * (self.first[i-1] + num_symbol[i-1])
        self.index = [0] * 16
        for i in range(1, 16):
            self.index[i] = self.index[i-1] + num_symbol[i-1]

    def decode(self, bitstream):
        l = 1
        a_list = []
        a = bitstream.GetBit()
        a_list.append(a)
        code = a
        while code > self.first[l-1]+self.num_symbol[l-1]-1:
            a = bitstream.GetBit()
            a_list.append(a)
            code = 2*code + a
            l += 1
        index = self.index[l-1] + code - self.first[l-1]
        sym = self.symbol_list[index]
        return sym

def get_CHT(data, obj):
    data_len = len(data)
    offset = 0
    cht_id = get_bytes(data, 1, offset)[0]
    offset += 1
    num_symbol = get_bytes(data, 16, offset)
    offset += 16
    total_symbol = sum(num_symbol)
    if data_len != offset + total_symbol:
        raise Exception("CHT data block error")
    symbol_list = get_bytes(data, total_symbol, offset)
    obj.huffman_tables[cht_id] = CHT(num_symbol, symbol_list)

def RemoveFF00(data):
    datapro = []
    i = 0
    while(True):
        b, bnext = unpack("BB",data[i:i+2])        
        if (b == 0xff):
            if (bnext != 0):
                break
            datapro.append(data[i])
            i+=2
        else:
            datapro.append(data[i])
            i+=1
    return datapro,i

def DecodeNumber(code, bits):
    l = 2**(code-1)
    if bits>=l:
        return bits
    else:
        return bits-(2*l-1)

def decode_block(data, cht_dc, cht_ac, dc_old):
    block = [0] * 64

    dc_code = cht_dc.decode(data)
    dc_bits = data.GetBitN(dc_code)
    dc_coeff = DecodeNumber(dc_code, dc_bits) + dc_old
    block[0] = dc_coeff 

    l = 1
    while l < 64:
        ac_code = cht_ac.decode(data)
        if ac_code == 0:
            break
        run_length = ac_code>>4
        bit_size = ac_code & 0x0F
        ac_bits = data.GetBitN(bit_size)
        ac_coeff = DecodeNumber(bit_size, ac_bits)
        l += run_length
        block[l] = ac_coeff
        l += 1

    return block, dc_coeff

def re_zigzag(block):
    zigzag = [
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ]
    zigzag_index = []
    for l in zigzag : zigzag_index += l
    re_zigzag_index = np.arange(64)
    re_zigzag_index[zigzag_index] = np.arange(64)
    target = np.arange(64)
    target[re_zigzag_index] = block
    return target
    
def get_scan_data(data, obj):
    block_data_Y = []
    block_data_Cb = []
    block_data_Cr = []
    dc_Y, dc_Cb, dc_Cr = 0, 0, 0
    for y in range(obj.height//(8*obj.MaxH)):
        for x in range(obj.width//(8*obj.MaxV)):
            # Y Component
            for _ in range(obj.sampling[1][0]*obj.sampling[1][1]):
                block, dc_Y = decode_block(data, obj.huffman_tables[0], obj.huffman_tables[16], dc_Y)
                block = np.array(block) * np.array(obj.quant[obj.quantMapping[1]])
                block = re_zigzag(block).reshape([8, 8])
                block = scipy.fft.idctn(block, type=2, norm='ortho')
                block_data_Y.append(block)
            # Cb Component
            for _ in range(obj.sampling[2][0]*obj.sampling[2][1]):
                block, dc_Cb = decode_block(data, obj.huffman_tables[1], obj.huffman_tables[17], dc_Cb)
                block = np.array(block) * np.array(obj.quant[obj.quantMapping[2]])
                block = re_zigzag(block).reshape([8, 8])
                block = scipy.fft.idctn(block, type=2, norm='ortho')
                block_data_Cb.append(block)
            # Cr Component
            for _ in range(obj.sampling[3][0]*obj.sampling[3][1]):
                block, dc_Cr = decode_block(data, obj.huffman_tables[1], obj.huffman_tables[17], dc_Cr)
                block = np.array(block) * np.array(obj.quant[obj.quantMapping[3]])
                block = re_zigzag(block).reshape([8, 8])
                block = scipy.fft.idctn(block, type=2, norm='ortho')
                block_data_Cr.append(block)
    return block_data_Y, block_data_Cb, block_data_Cr

def DrawImage444(img):
    x_limit = img.height//8
    y_limit = img.width//8
    img.result = np.zeros([img.height, img.width, 3], dtype=np.uint8)
    for x in range(x_limit):
        for y in range(y_limit):
            img.result[x*8:x*8+8, y*8:y*8+8, 0] = img.block_data_Y[x*y_limit+y] + 128
            img.result[x*8:x*8+8, y*8:y*8+8, 1] = img.block_data_Cb[x*y_limit+y] + 128
            img.result[x*8:x*8+8, y*8:y*8+8, 2] = img.block_data_Cr[x*y_limit+y] + 128
    img.result[img.result>255] = 255
    img.result[img.result<0] = 0
    image = Image.fromarray(img.result, 'YCbCr')
    image = image.convert('RGB')
    image.show()

def upsample(x):
    m, n = x.shape
    x = np.tile(x[:, None, :, None], [1,2,1,2]).reshape([2*m, 2*n])
    return x

def DrawImage420(img):
    x_limit = img.height//16
    y_limit = img.width//16
    img.result = np.zeros([img.height, img.width, 3], dtype=np.uint8)
    for x in range(x_limit):
        for y in range(y_limit):
            img.result[x*16:x*16+8, y*16:y*16+8, 0] = img.block_data_Y[4*(x*y_limit+y)+0] + 128
            img.result[x*16:x*16+8, y*16+8:y*16+16, 0] = img.block_data_Y[4*(x*y_limit+y)+1] + 128
            img.result[x*16+8:x*16+16, y*16:y*16+8, 0] = img.block_data_Y[4*(x*y_limit+y)+2] + 128
            img.result[x*16+8:x*16+16, y*16+8:y*16+16, 0] = img.block_data_Y[4*(x*y_limit+y)+3] + 128
            img.result[x*16:x*16+16, y*16:y*16+16, 1] = upsample(img.block_data_Cb[x*y_limit+y]) + 128
            img.result[x*16:x*16+16, y*16:y*16+16, 2] = upsample(img.block_data_Cr[x*y_limit+y]) + 128
    img.result[img.result>255] = 255
    img.result[img.result<0] = 0
    image = Image.fromarray(img.result, 'YCbCr')
    image = image.convert('RGB')
    image.show()

def DrawImage(img):
    if img.MaxH==2:
        DrawImage420(img)
    elif img.MaxH==1:
        DrawImage444(img)
    else:
        raise NotImplementedError