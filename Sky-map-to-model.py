#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sky 地图转模型工具

作者: 十二
作者联系方式:
q:3787533101
邮箱:3787533101@qq.com
项目链接:https://github.com/skyIshier/Sky-map-to-model

依赖：pip install lz4 trimesh numpy
"""

import os
import sys
import json
import struct
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import lz4.block
import numpy as np
import trimesh

# =============================================================================
# 第一部分：TGCL 解析器（完整保留，用于解析 Objects.level.bin）
# =============================================================================

# ---------- 翻译表 ----------
CLASS_NAMES_ZH = {}
PROPERTY_NAMES_ZH = {}

def translate_class_name(name):
    zh = CLASS_NAMES_ZH.get(name, "")
    if zh:
        return f"{name}（{zh}）"
    return name

def translate_property_name(name):
    zh = PROPERTY_NAMES_ZH.get(name, "")
    if zh:
        return f"{name}（{zh}）"
    return name

def orig_name(name):
    if "（" in name and name.endswith("）"):
        return name.split("（")[0]
    if " (" in name and name.endswith(")"):
        return name.split(" (")[0]
    return name

def float_from_u32(u32: int) -> float:
    return struct.unpack('f', struct.pack('I', u32))[0]

def u32_from_float(f: float) -> int:
    return struct.unpack('I', struct.pack('f', f))[0]

def double_from_raw(raw_bytes: bytes) -> float:
    return struct.unpack('d', raw_bytes)[0]

NAN_SENTINEL = 0xFFFFFFF6

def should_read_uint32_as_integer(property_name: str) -> bool:
    real = orig_name(property_name)
    return real == "bstGuid" or "BstGuid" in real

def ends_with(value: str, suffix: str) -> bool:
    return value.endswith(suffix)

def is_nan_clump_string(value: str) -> bool:
    v = value.strip().lower()
    return v in ("", "-nan", "nan", "null")

def is_nan_numeric_text(value: str) -> bool:
    v = value.strip().lower()
    return v in ("-nan", "nan")

def _get_original_bin_candidates(json_file_path: str) -> list:
    base, ext = os.path.splitext(json_file_path)
    candidates = [base]
    for suffix in (".parsed", ".parser"):
        if ends_with(base, suffix):
            candidates.append(base[:-len(suffix)])
    return candidates

class ClassDef:
    __slots__ = ('classPropertyNameOffset', 'classPropertyStartingIndex',
                 'classPropertyCount', 'className')
    def __init__(self):
        self.classPropertyNameOffset = 0
        self.classPropertyStartingIndex = 0
        self.classPropertyCount = 0
        self.className = ""
    @staticmethod
    def read(stream):
        c = ClassDef()
        data = stream.read(12)
        c.classPropertyNameOffset, c.classPropertyStartingIndex, c.classPropertyCount = struct.unpack('<III', data)
        return c

class PropertyDef:
    __slots__ = ('propertyType', 'propertyNameOffset', 'objectByteSize',
                 'arrayIndex', 'propertyName')
    def __init__(self):
        self.propertyType = 0
        self.propertyNameOffset = 0
        self.objectByteSize = 0
        self.arrayIndex = 0
        self.propertyName = ""
    @staticmethod
    def read(stream):
        p = PropertyDef()
        data = stream.read(16)
        p.propertyType, p.propertyNameOffset, p.objectByteSize, p.arrayIndex = struct.unpack('<IIII', data)
        return p

class BSTHeader:
    __slots__ = ('magic', 'version', 'classLength', 'propertyCount',
                 'BSTNodeCount', 'objectPtrCount', 'classOffset',
                 'propertyOffset', 'PropertyNameOffset', 'BSTNodeOffset', 'FileSize')
    def __init__(self, stream):
        data = stream.read(44)
        (self.magic, self.version, self.classLength, self.propertyCount,
         self.BSTNodeCount, self.objectPtrCount, self.classOffset,
         self.propertyOffset, self.PropertyNameOffset, self.BSTNodeOffset,
         self.FileSize) = struct.unpack('<4sIIIIIIIIII', data)
    @property
    def magic_str(self):
        return self.magic.decode('ascii', errors='replace')

def _read_cstring(stream) -> str:
    result = bytearray()
    while True:
        ch = stream.read(1)
        if not ch or ch == b'\x00':
            break
        result.extend(ch)
    return result.decode('utf-8', errors='replace')

def _read_classes(stream, header: BSTHeader) -> list:
    stream.seek(header.classOffset)
    return [ClassDef.read(stream) for _ in range(header.classLength)]

def _read_all_properties(stream, header: BSTHeader, classes: list) -> list:
    stream.seek(header.propertyOffset)
    all_props = []
    for cls in classes:
        props = []
        if cls.classPropertyCount > 0:
            stream.seek(header.propertyOffset + cls.classPropertyStartingIndex * 16)
            for _ in range(cls.classPropertyCount):
                props.append(PropertyDef.read(stream))
            for p in props:
                stream.seek(header.PropertyNameOffset + p.propertyNameOffset)
                p.propertyName = _read_cstring(stream)
        all_props.append(props)
        stream.seek(header.PropertyNameOffset + cls.classPropertyNameOffset)
        cls.className = _read_cstring(stream)
    return all_props

def _make_original_key(node, cls, prop):
    return f"{node}\x1F{cls}\x1F{prop}"

def load_original_string_pool_order(json_file_path: str) -> list:
    candidates = _get_original_bin_candidates(json_file_path)
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, 'rb') as f:
                header = BSTHeader(f)
                if header.magic_str != 'TGCL':
                    continue
                f.seek(header.PropertyNameOffset)
                names = []
                while f.tell() < header.BSTNodeOffset:
                    name = _read_cstring(f)
                    if name:
                        names.append(name)
                if names:
                    return names
        except:
            pass
    return []

def load_original_top_level_u32(json_file_path: str) -> dict:
    candidates = _get_original_bin_candidates(json_file_path)
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, 'rb') as f:
                header = BSTHeader(f)
                if header.magic_str != 'TGCL':
                    continue
                classes = _read_classes(f, header)
                all_props = _read_all_properties(f, header, classes)
                f.seek(header.BSTNodeOffset)
                out = {}
                for _ in range(header.BSTNodeCount):
                    index = struct.unpack('<I', f.read(4))[0]
                    bst_name = _read_cstring(f)
                    _capture_u32(f, bst_name, index, all_props, classes, out, True)
                if out:
                    return out
        except:
            pass
    return {}

def _capture_u32(stream, node_name, class_idx, all_props, classes, out, top):
    if class_idx >= len(all_props):
        return
    for prop in all_props[class_idx]:
        try:
            if prop.propertyType == 0 and prop.objectByteSize == 4:
                raw = struct.unpack('<I', stream.read(4))[0]
                if top:
                    key = _make_original_key(node_name, classes[class_idx].className, prop.propertyName)
                    out[key] = raw
            elif prop.propertyType == 0:
                skip = prop.objectByteSize if prop.objectByteSize > 0 else 4
                stream.seek(skip, 1)
            elif prop.propertyType == 1:
                _read_cstring(stream)
            elif prop.propertyType == 2:
                stream.seek(4, 1)
            elif prop.propertyType == 3:
                count = min(struct.unpack('<I', stream.read(4))[0], 100000)
                if prop.arrayIndex != 0xFFFFFFFF:
                    for _ in range(count):
                        _capture_u32(stream, node_name, prop.arrayIndex, all_props, classes, out, False)
                else:
                    stream.seek(count * 4, 1)
            else:
                stream.seek(max(prop.objectByteSize, 4), 1)
        except:
            break

def bin_to_json(input_path: str, translate: bool = True) -> OrderedDict:
    """将 .bin 文件解析为 OrderedDict JSON 结构"""
    with open(input_path, 'rb') as f:
        header = BSTHeader(f)
        if header.magic_str != 'TGCL':
            raise ValueError("不是有效的 TGCL 文件")

        classes = _read_classes(f, header)
        all_props = _read_all_properties(f, header, classes)

        j = OrderedDict()
        j['version'] = header.version
        j['MemorySize'] = str(header.objectPtrCount)
        j['classes'] = OrderedDict()
        j['BSTNodes'] = OrderedDict()

        for i, cls in enumerate(classes):
            cls_key = translate_class_name(cls.className) if translate else cls.className
            if not all_props[i]:
                j['classes'][cls_key] = None
            else:
                meta = OrderedDict()
                for prop in all_props[i]:
                    prop_key = translate_property_name(prop.propertyName) if translate else prop.propertyName
                    meta[prop_key] = OrderedDict([
                        ('propertyType', prop.propertyType),
                        ('objectByteSize', prop.objectByteSize),
                        ('arrayIndex', prop.arrayIndex)
                    ])
                j['classes'][cls_key] = meta

        f.seek(header.BSTNodeOffset)
        for _ in range(header.BSTNodeCount):
            index = struct.unpack('<I', f.read(4))[0]
            bst_name = _read_cstring(f)
            node = OrderedDict()
            _read_class_data(f, node, index, all_props, classes, translate)
            j['BSTNodes'][bst_name] = node

        bst_names = list(j['BSTNodes'].keys())
        for bst_name, node in j['BSTNodes'].items():
            if isinstance(node, dict):
                for cls_name, cls_data in node.items():
                    real_cls = orig_name(cls_name)
                    meta = j['classes'].get(real_cls, {})
                    if not meta:
                        for k in j['classes']:
                            if orig_name(k) == real_cls:
                                meta = j['classes'][k]
                                break
                    if isinstance(meta, dict) and isinstance(cls_data, dict):
                        _resolve_clump_refs(cls_data, j['classes'], meta, classes, bst_names)

        return j

def _read_class_data(stream, json_data, index, all_props, classes, translate=True):
    if index >= len(classes) or index >= len(all_props):
        return
    cls = classes[index]
    cls_key = translate_class_name(cls.className) if translate else cls.className
    if cls_key not in json_data:
        json_data[cls_key] = OrderedDict()
    target = json_data[cls_key]

    for prop in all_props[index]:
        try:
            ptype = prop.propertyType
            psize = prop.objectByteSize
            prop_key = translate_property_name(prop.propertyName) if translate else prop.propertyName

            if ptype == 0:
                if psize == 1:
                    v = struct.unpack('<B', stream.read(1))[0]
                    target[prop_key] = OrderedDict([('_raw_uint8', v), ('_value', bool(v) if v <= 1 else f"0x{v:02X}")])
                elif psize == 2:
                    v = struct.unpack('<H', stream.read(2))[0]
                    target[prop_key] = OrderedDict([('_raw_uint16', v), ('_value', v)])
                elif psize == 4:
                    raw = struct.unpack('<I', stream.read(4))[0]
                    if should_read_uint32_as_integer(prop.propertyName):
                        target[prop_key] = OrderedDict([('_raw_uint32', raw), ('_value', str(raw))])
                    else:
                        fval = float_from_u32(raw)
                        target[prop_key] = OrderedDict([('_raw_uint32', raw), ('_value', str(fval))])
                elif psize == 8:
                    raw_bytes = stream.read(8)
                    dval = double_from_raw(raw_bytes)
                    target[prop_key] = OrderedDict([('_raw_bytes_hex', raw_bytes.hex()), ('_value', str(dval))])
                elif psize == 10:
                    raw_bytes = stream.read(10)
                    dval = double_from_raw(raw_bytes[:8])
                    target[prop_key] = OrderedDict([('_raw_bytes_hex', raw_bytes.hex()), ('_value', str(dval))])
                elif psize == 16:
                    vals = struct.unpack('<ffff', stream.read(16))
                    target[prop_key] = OrderedDict([('_raw_floats', [str(v) for v in vals]), ('_value', [str(v) for v in vals])])
                elif psize == 64:
                    vals = struct.unpack('<16f', stream.read(64))
                    mat = [[str(vals[i*4+j]) for j in range(4)] for i in range(4)]
                    target[prop_key] = OrderedDict([('_raw_floats', [str(v) for v in vals]), ('_value', mat)])
                else:
                    raw_bytes = stream.read(psize)
                    target[prop_key] = OrderedDict([('_raw_bytes_hex', raw_bytes.hex()), ('_value', f"[大小 {psize} 字节]")])
            elif ptype == 1:
                target[prop_key] = _read_cstring(stream)
            elif ptype == 2:
                raw = struct.unpack('<I', stream.read(4))[0]
                target[prop_key] = OrderedDict([('_raw_uint32', raw), ('_is_clump', True)])
            elif ptype == 3:
                count = min(struct.unpack('<I', stream.read(4))[0], 100000)
                if prop.arrayIndex != 0xFFFFFFFF:
                    arr = []
                    for _ in range(count):
                        elem = OrderedDict()
                        _read_class_data(stream, elem, prop.arrayIndex, all_props, classes, translate)
                        arr.append(elem)
                    target[prop_key] = arr
                else:
                    clump_data = []
                    for _ in range(count):
                        raw = struct.unpack('<I', stream.read(4))[0]
                        clump_data.append(OrderedDict([('_raw_uint32', raw), ('_is_clump', True)]))
                    target[prop_key] = OrderedDict([('_array_count', count), ('_elements', clump_data)])
            else:
                raw_bytes = stream.read(max(psize, 4))
                target[prop_key] = OrderedDict([('_raw_bytes_hex', raw_bytes.hex()), ('_value', f"[未知类型 {psize} 字节]")])
        except Exception as e:
            print(f"警告: 读取属性 '{prop.propertyName}' 失败: {e}")
            break

def _resolve_clump_refs(class_data, all_meta, class_meta, classes, bst_names):
    for prop_name, meta in class_meta.items():
        if not isinstance(meta, dict):
            continue
        ptype = meta.get('propertyType')
        array_idx = meta.get('arrayIndex')
        real_prop = orig_name(prop_name)

        if ptype == 2:
            val = class_data.get(prop_name)
            if isinstance(val, dict) and val.get('_is_clump'):
                raw = val['_raw_uint32']
                fv = float_from_u32(raw)
                if str(fv) == 'nan':
                    val['_clump_name'] = 'nan'
                elif raw < len(bst_names):
                    val['_clump_name'] = bst_names[raw]
                else:
                    val['_clump_name'] = str(raw)
        elif ptype == 3 and array_idx is not None and array_idx != 0xFFFFFFFF and array_idx < len(classes):
            arr = class_data.get(prop_name)
            if isinstance(arr, list):
                nested_cls = classes[array_idx].className
                nested_meta = all_meta.get(nested_cls, {})
                if not nested_meta:
                    for k in all_meta:
                        if orig_name(k) == nested_cls:
                            nested_meta = all_meta[k]
                            break
                for elem in arr:
                    if isinstance(elem, dict):
                        _resolve_clump_refs(elem, all_meta, nested_meta, classes, bst_names)
        elif ptype == 3:
            obj = class_data.get(prop_name)
            if isinstance(obj, dict) and '_elements' in obj:
                for elem in obj['_elements']:
                    if isinstance(elem, dict) and elem.get('_is_clump'):
                        raw = elem['_raw_uint32']
                        fv = float_from_u32(raw)
                        if str(fv) == 'nan':
                            elem['_clump_name'] = 'nan'
                        elif raw < len(bst_names):
                            elem['_clump_name'] = bst_names[raw]
                        else:
                            elem['_clump_name'] = str(raw)

# ---------- 从 JSON 数据中提取 LevelMesh 和标记点 ----------
def extract_level_meshes_from_json(json_data: dict) -> List[Dict]:
    """从 bintojson 输出的 JSON 中提取所有 LevelMesh 实例"""
    out = []
    seen = set()
    nodes = json_data.get('BSTNodes', {})
    classes = json_data.get('classes', {})

    class_meta = {}
    for cls_name, meta in classes.items():
        if meta is None:
            class_meta[cls_name] = {}
        else:
            class_meta[cls_name] = meta

    for node_name, node_data in nodes.items():
        if not isinstance(node_data, dict):
            continue
        for cls_name, fields in node_data.items():
            if not isinstance(fields, dict):
                continue
            res_name = None
            for key in ['resourceName', 'mesh', 'meshName']:
                if key in fields and isinstance(fields[key], str):
                    res_name = fields[key]
                    break
            if not res_name:
                continue
            tf = None
            if 'transform' in fields:
                tf_val = fields['transform']
                if isinstance(tf_val, dict) and '_raw_floats' in tf_val:
                    fl = tf_val['_raw_floats']
                    if len(fl) >= 16:
                        tf = [float(x) for x in fl[:16]]
                elif isinstance(tf_val, list) and len(tf_val) >= 16:
                    tf = [float(x) for x in tf_val[:16]]
            if tf is None:
                continue
            key = res_name + '|' + ','.join([str(x) for x in tf])
            if key in seen:
                continue
            seen.add(key)
            shader_name = fields.get('shaderName', '') if isinstance(fields.get('shaderName'), str) else ''
            diffuse_tex = ''
            norm_tex = ''
            diffuse2_tex = ''
            light_tex = ''
            diffuse2_offset = None
            diffuse_color = None
            base_color = None
            if 'shaderParams' in fields:
                sp = fields['shaderParams']
                if isinstance(sp, list):
                    for item in sp:
                        if isinstance(item, dict):
                            un = item.get('uniformName')
                            tv = item.get('texValue')
                            if un in ('u_diffuse1Tex', 'u_diffuseTex'):
                                diffuse_tex = tv
                            elif un == 'u_normTex':
                                norm_tex = tv
                            elif un == 'u_diffuse2Tex':
                                diffuse2_tex = tv
                            elif un == 'u_lightTex':
                                light_tex = tv
                            elif un == 'u_diffuse2TexOffset' and 'vecValue' in item:
                                diffuse2_offset = item['vecValue']
                            elif un == 'u_diffuseColor' and 'vecValue' in item:
                                diffuse_color = item['vecValue']
            b = tf
            matrix = [
                b[0], b[4], b[8], b[12],
                b[1], b[5], b[9], b[13],
                b[2], b[6], b[10], b[14],
                0, 0, 0, 1
            ]
            out.append({
                'name': node_name,
                'resourceName': res_name,
                'shaderName': shader_name,
                'diffuseTex': diffuse_tex,
                'normTex': norm_tex,
                'diffuse2Tex': diffuse2_tex,
                'lightTex': light_tex,
                'diffuse2Offset': diffuse2_offset,
                'diffuseColor': diffuse_color,
                'baseColor': base_color,
                'matrix': matrix
            })
    return out

def extract_level_markers_from_json(json_data: dict) -> List[Dict]:
    """从 bintojson 输出的 JSON 中提取所有标记点"""
    MARKER_DEFS = [
        ('CandleObject', '烛火', '#ff9a3c'),
        ('WingBuff', '光翼', '#5fd3ff'),
        ('Pickup', '拾取物', '#ffd54a'),
        ('PickupEmitter', '蜡堆', '#ffb74a'),
        ('MeditationArea', '冥想点', '#b98cff'),
        ('Portal', '传送门', '#00e0c0'),
        ('Checkpoint', '存档点', '#7dff8a'),
        ('ConstellationMarker', '星座点', '#ffe680'),
        ('MapShrine', '地图石', '#c0c8d0'),
        ('Npc', 'NPC', '#ff8fd0'),
        ('LevelLink', '关卡门', '#a0b4ff'),
        ('StarFragment', '星之碎片', '#ffe680'),
        ('Collectible', '收集品', '#ffd54a'),
        ('Flame', '火焰', '#ff7a3c'),
        ('SoundEmitter', '音效点', '#9ad0ff'),
        ('DisplayText', '显示文字', '#d0d8e0'),
        ('PointLight', '点光源', '#ffef9e'),
        ('ConstellationGate', '星座门', '#c8a8ff'),
        ('StreamingCrystal', '回忆水晶', '#8ad0ff'),
    ]
    groups = []
    nodes = json_data.get('BSTNodes', {})

    for type_name, label, color in MARKER_DEFS:
        points = []
        for node_name, node_data in nodes.items():
            if not isinstance(node_data, dict):
                continue
            if type_name not in node_data:
                continue
            fields = node_data[type_name]
            if not isinstance(fields, dict):
                continue
            pos = None
            if 'transform' in fields:
                tf = fields['transform']
                if isinstance(tf, dict) and '_raw_floats' in tf:
                    fl = tf['_raw_floats']
                    if len(fl) >= 16:
                        pos = [float(fl[12]), float(fl[13]), float(fl[14])]
                elif isinstance(tf, list) and len(tf) >= 16:
                    pos = [float(tf[12]), float(tf[13]), float(tf[14])]
            if pos is None:
                if 'pos' in fields and isinstance(fields['pos'], (list, tuple)) and len(fields['pos']) >= 3:
                    pos = [float(x) for x in fields['pos'][:3]]
            if pos and any(pos):
                points.append({'pos': pos, 'label': label})
        if points:
            groups.append({'key': type_name, 'label': label, 'color': color, 'points': points})
    return groups

# =============================================================================
# 第二部分：通用辅助函数（读取 .mesh 等）
# =============================================================================

def read_u32(data: bytes, offset: int, le: bool = True) -> int:
    return struct.unpack('<I' if le else '>I', data[offset:offset+4])[0]

def read_f32(data: bytes, offset: int, le: bool = True) -> float:
    return struct.unpack('<f' if le else '>f', data[offset:offset+4])[0]

def read_string(data: bytes, offset: int, max_len: int) -> str:
    end = offset
    while end < len(data) and end - offset < max_len and data[end] != 0:
        end += 1
    return data[offset:end].decode('utf-8', errors='ignore')

def half_to_float(h: int) -> float:
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1f
    m = h & 0x3ff
    if e == 0:
        if m == 0:
            return -0.0 if s else 0.0
        while (m & 0x400) == 0:
            m <<= 1
            e -= 1
        e += 1
        m &= 0x3ff
    elif e == 31:
        return float('nan') if m else (float('-inf') if s else float('inf'))
    bits = (s << 31) | ((e + 112) << 23) | (m << 13)
    return struct.unpack('f', struct.pack('I', bits))[0]

# =============================================================================
# LZ4 解压
# =============================================================================
def lz4_decompress(src: bytes, expected_max: int) -> bytes:
    if not src:
        return b''
    max_out = min(expected_max if expected_max > 0 else 0xC00000, 128 * 1024 * 1024)
    try:
        dst = lz4.block.decompress(src, uncompressed_size=max_out)
        return dst
    except Exception as e:
        raise RuntimeError(f"LZ4解压失败: {e}")

# =============================================================================
# Meshopt 解码器（完整，用于 .mesh 文件）
# =============================================================================
K_VERTEX_HEADER = 0xA0
K_VERTEX_BLOCK_SIZE_BYTES = 8192
K_VERTEX_BLOCK_MAX_SIZE = 256
K_BYTE_GROUP_SIZE = 16

REVERSE_BITS8 = [0] * 256
for i in range(256):
    v = i
    r = 0
    for _ in range(8):
        r = ((r << 1) | (v & 1)) & 0xFF
        v >>= 1
    REVERSE_BITS8[i] = r

def get_vertex_block_size(vertex_size: int) -> int:
    result = (K_VERTEX_BLOCK_SIZE_BYTES // vertex_size) & ~(K_BYTE_GROUP_SIZE - 1)
    if result < K_VERTEX_BLOCK_MAX_SIZE:
        return result
    return K_VERTEX_BLOCK_MAX_SIZE

def decode_bytes_group(data: bytes, pos: int, out: bytearray, out_off: int, bits: int) -> int:
    if bits == 0:
        for i in range(16):
            out[out_off + i] = 0
        return pos
    if bits == 8:
        out[out_off:out_off+16] = data[pos:pos+16]
        return pos + 16
    sentinel = (1 << bits) - 1
    byte_size = 8 // bits
    fixed_count = 16 // byte_size
    var_pos = pos + fixed_count
    idx = out_off
    for fb in range(fixed_count):
        b = data[pos + fb]
        if bits == 1:
            b = REVERSE_BITS8[b]
        for _ in range(byte_size):
            enc = (b >> (8 - bits)) & 0xFF
            b = (b << bits) & 0xFF
            if enc == sentinel:
                out[idx] = data[var_pos]
                var_pos += 1
            else:
                out[idx] = enc
            idx += 1
    return var_pos

def decode_bytes(data: bytes, pos: int, buffer_size: int, bits_table: List[int], out: bytearray) -> int:
    num_groups = (buffer_size + 15) // 16
    header_size = (num_groups + 3) // 4
    header_base = pos
    pos += header_size
    for g in range(num_groups):
        bitsk = ((data[header_base + (g // 4)] >> ((g % 4) * 2)) & 3)
        pos = decode_bytes_group(data, pos, out, g * 16, bits_table[bitsk])
    return pos

def decode_deltas_u8(planes: List[bytes], result: bytearray, base: int, vertex_count: int, vertex_size: int,
                     last_vertex: List[int], k: int):
    for kb in range(4):
        plane = planes[kb]
        p = last_vertex[k + kb] & 0xFF
        off = base + kb
        for i in range(vertex_count):
            v0 = plane[i] & 0xFF
            zigzag = ((v0 & 1) != 0) * 0xFF ^ (v0 >> 1)
            v = (zigzag + p) & 0xFF
            result[off] = v
            p = v
            off += vertex_size

def decode_deltas_u16(planes: List[bytes], result: bytearray, base: int, vertex_count: int, vertex_size: int,
                      last_vertex: List[int], k: int):
    for kb in range(0, 4, 2):
        p = (last_vertex[k + kb] & 0xFF) | ((last_vertex[k + kb + 1] & 0xFF) << 8)
        off = base + kb
        plane0 = planes[kb]
        plane1 = planes[kb+1]
        for i in range(vertex_count):
            v0 = (plane0[i] & 0xFF) | ((plane1[i] & 0xFF) << 8)
            zigzag = ((v0 & 1) != 0) * 0xFFFF ^ (v0 >> 1)
            v = (zigzag + p) & 0xFFFF
            result[off] = v & 0xFF
            result[off+1] = (v >> 8) & 0xFF
            p = v
            off += vertex_size

def decode_deltas_u32_xor(planes: List[bytes], result: bytearray, base: int, vertex_count: int, vertex_size: int,
                          last_vertex: List[int], k: int, rot: int):
    p = ((last_vertex[k] & 0xFF) |
         ((last_vertex[k+1] & 0xFF) << 8) |
         ((last_vertex[k+2] & 0xFF) << 16) |
         ((last_vertex[k+3] & 0xFF) << 24)) & 0xFFFFFFFF
    off = base
    p0, p1, p2, p3 = planes[0], planes[1], planes[2], planes[3]
    if rot == 0:
        for i in range(vertex_count):
            cur = ((p0[i] & 0xFF) | ((p1[i] & 0xFF) << 8) | ((p2[i] & 0xFF) << 16) | ((p3[i] & 0xFF) << 24)) & 0xFFFFFFFF
            v = (cur ^ p) & 0xFFFFFFFF
            result[off] = v & 0xFF
            result[off+1] = (v >> 8) & 0xFF
            result[off+2] = (v >> 16) & 0xFF
            result[off+3] = (v >> 24) & 0xFF
            p = v
            off += vertex_size
    else:
        rshift = 32 - rot
        for i in range(vertex_count):
            cur = ((p0[i] & 0xFF) | ((p1[i] & 0xFF) << 8) | ((p2[i] & 0xFF) << 16) | ((p3[i] & 0xFF) << 24)) & 0xFFFFFFFF
            v = (((cur << rot) | (cur >> rshift)) ^ p) & 0xFFFFFFFF
            result[off] = v & 0xFF
            result[off+1] = (v >> 8) & 0xFF
            result[off+2] = (v >> 16) & 0xFF
            result[off+3] = (v >> 24) & 0xFF
            p = v
            off += vertex_size

def decode_vertex_block(data: bytes, pos: int, result: bytearray, vertex_offset: int, vertex_count: int,
                        vertex_size: int, last_vertex: List[int], channels: Optional[bytes], version: int) -> int:
    vertex_count_aligned = (vertex_count + 15) & ~15
    control_size = 0 if version == 0 else (vertex_size // 4)
    control_base = pos
    pos += control_size
    planes = [None, None, None, None]
    for k in range(0, vertex_size, 4):
        ctrl_byte = 0 if version == 0 else data[control_base + (k // 4)]
        for j in range(4):
            ctrl = (ctrl_byte >> (j * 2)) & 3
            if ctrl == 3:
                planes[j] = data[pos:pos+vertex_count]
                pos += vertex_count
            elif ctrl == 2:
                planes[j] = bytearray(vertex_count)
            else:
                bits_table = [0, 2, 4, 8] if version == 0 else ([0, 1, 2, 4] if ctrl == 0 else [1, 2, 4, 8])
                out = bytearray(vertex_count_aligned)
                pos = decode_bytes(data, pos, vertex_count_aligned, bits_table, out)
                planes[j] = out
        channel = 0 if version == 0 else (channels[k // 4] if channels else 0)
        ctype = channel & 3
        base = vertex_offset * vertex_size + k
        if ctype == 0:
            decode_deltas_u8(planes, result, base, vertex_count, vertex_size, last_vertex, k)
        elif ctype == 1:
            decode_deltas_u16(planes, result, base, vertex_count, vertex_size, last_vertex, k)
        else:
            rot = (32 - ((channel >> 4) & 31)) & 31
            decode_deltas_u32_xor(planes, result, base, vertex_count, vertex_size, last_vertex, k, rot)
    last_start = vertex_offset * vertex_size + (vertex_count - 1) * vertex_size
    last_vertex[:] = list(result[last_start:last_start+vertex_size])
    return pos

def meshopt_decode_vertex_buffer(vertex_count: int, vertex_size: int, data: bytes) -> bytes:
    if vertex_size % 4 != 0:
        raise ValueError("vertex size must be multiple of 4")
    if len(data) < 1:
        raise ValueError("empty meshopt data")
    header = data[0] & 0xFF
    if (header & 0xF0) != K_VERTEX_HEADER:
        raise ValueError(f"meshopt header mismatch: 0x{header:02X}")
    version = header & 0x0F
    if version > 1:
        raise ValueError(f"unsupported meshopt version: {version}")
    tail_size = vertex_size + (0 if version == 0 else (vertex_size // 4))
    tail_size_min = 32 if version == 0 else 24
    tail_size = max(tail_size, tail_size_min)
    if len(data) < 1 + tail_size:
        raise ValueError("meshopt data too short")
    tail_start = len(data) - tail_size
    last_vertex = list(data[tail_start:tail_start+vertex_size])
    channels = None
    if version != 0:
        channels = data[tail_start+vertex_size:tail_start+vertex_size+(vertex_size//4)]
    vertex_block_size = get_vertex_block_size(vertex_size)
    result = bytearray(vertex_count * vertex_size)
    pos = 1
    vertex_offset = 0
    while vertex_offset < vertex_count:
        block_size = min(vertex_block_size, vertex_count - vertex_offset)
        pos = decode_vertex_block(data, pos, result, vertex_offset, block_size, vertex_size, last_vertex, channels, version)
        vertex_offset += block_size
    return bytes(result)

# =============================================================================
# .mesh 解析（全版本，整合自 fmt_mesh 并适配为字典返回）
# =============================================================================

def _u32(d, o):
    return struct.unpack_from('<I', d, o)[0]

def _f32(d, o):
    return struct.unpack_from('<f', d, o)[0]

def _vec3(d, o):
    return (_f32(d, o), _f32(d, o+4), _f32(d, o+8))

def has_skeleton_flag(data):
    if len(data) <= 0x48:
        return 0
    return data[0x48]

def build_bones_from_block(block, bone_count, start):
    bones = []
    p = start
    for x in range(bone_count):
        if p + 132 > len(block):
            break
        name_raw = block[p:p+64].split(b'\x00')[0]
        name = name_raw.decode('ascii', errors='ignore') if name_raw else "bone_{}".format(x)
        p += 64
        mat = list(struct.unpack('<16f', block[p:p+64]))
        p += 64
        parent_idx = struct.unpack_from('<I', block, p)[0] - 1
        p += 4
        bones.append({'name': name, 'parent': parent_idx, 'matrix': mat})
    return bones

def parse_tail_bones(tail):
    if len(tail) < 85:
        return []
    bone_count = _u32(tail, 68)
    if bone_count <= 0 or bone_count > 4096:
        return []
    return build_bones_from_block(tail, bone_count, 85)

def parse_zippos_payload(dr, has_skin, bones):
    p = 4
    p += 12 + 12
    aabb_min = _vec3(dr, p); p += 12
    aabb_max = _vec3(dr, p); p += 12
    quant_min = [_f32(dr, p + i*4) for i in range(8)]; p += 32
    quant_max = [_f32(dr, p + i*4) for i in range(8)]; p += 32

    shared = _u32(dr, p); p += 4
    total = _u32(dr, p); p += 4
    is_idx32 = _u32(dr, p) != 0; p += 4
    num_points = _u32(dr, p); p += 4
    prop11 = _u32(dr, p); p += 4
    prop12 = _u32(dr, p); p += 4
    prop13 = _u32(dr, p); p += 4
    prop14 = _u32(dr, p); p += 4
    load_norms = dr[p] != 0; p += 1
    load_info2 = dr[p] != 0; p += 1
    p += 1
    skip_mesh_pos = _u32(dr, p); p += 4
    skip_uvs = _u32(dr, p); p += 4
    flag3 = _u32(dr, p); p += 4
    p += 0x10

    face_count = total // 3
    idx_unit = 4 if is_idx32 else 2

    if skip_mesh_pos == 0:
        inline_verts = dr[p : p + shared*16]
        p += shared * 16
    else:
        inline_verts = b''
    if load_norms:
        p += shared * 4
    inline_uv_off = None
    if skip_uvs == 0:
        inline_uv_off = p
        p += shared * 16
    wbuf = None
    if has_skin:
        wbuf = dr[p : p + shared*8]
        p += shared * 8
    ibuf = dr[p : p + face_count*3*idx_unit]
    p += face_count * 3 * idx_unit

    if load_info2:
        p += total * idx_unit
    if num_points > 0:
        p += shared * idx_unit
    if prop11 > 0:
        p += shared * idx_unit
    if prop12 > 0:
        p += prop12 * idx_unit
    if prop13 > 0:
        p += prop13 * 4
    if prop14 > 0:
        p += prop14 * (8 if is_idx32 else 4)
    p += face_count * 4

    if skip_mesh_pos > 0:
        ax, ay, az = aabb_min
        sx = aabb_max[0] - ax
        sy = aabb_max[1] - ay
        sz = aabb_max[2] - az
        positions = []
        for i in range(shared):
            pk = _u32(dr, p + i*4)
            qz = pk & 0x3FF
            qy = (pk >> 10) & 0x3FF
            qx = (pk >> 20) & 0x3FF
            positions.extend([ax + (qx / 1023.0) * sx,
                              ay + (qy / 1023.0) * sy,
                              az + (qz / 1023.0) * sz])
        p += shared * 4 + shared
    else:
        positions = []
        for i in range(shared):
            off = i * 16
            positions.extend(struct.unpack_from('<3f', inline_verts, off))

    if skip_uvs > 0:
        umin, vmin = quant_min[0], quant_min[1]
        usz = quant_max[0] - umin
        vsz = quant_max[1] - vmin
        uvs = []
        for i in range(shared):
            off = p + i*4
            u_hi, v_hi, u_lo, v_lo = dr[off], dr[off+1], dr[off+2], dr[off+3]
            un = ((u_hi << 8) | u_lo) / 65535.0
            vn = ((v_hi << 8) | v_lo) / 65535.0
            uvs.extend([umin + un * usz, vmin + vn * vsz])
        p += shared * 4
    else:
        if inline_uv_off is not None:
            uvbuf = dr[inline_uv_off : inline_uv_off + shared*16]
            uvs = []
            for i in range(shared):
                off = i * 16
                uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off+4)[0]))
                uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off+6)[0]))
        else:
            uvs = []

    idx_fmt = '<I' if is_idx32 else '<H'
    indices = list(struct.unpack(f'{len(ibuf)//(4 if is_idx32 else 2)}{idx_fmt}', ibuf))

    norms = None
    if load_norms:
        norms = []

    bone_indices = None
    bone_weights = None
    if has_skin and wbuf and bones:
        bone_indices = [0.0] * (shared * 4)
        bone_weights = [0.0] * (shared * 4)
        for i in range(shared):
            off = i * 8
            for j in range(4):
                bone_indices[i*4 + j] = wbuf[off + j]
                bone_weights[i*4 + j] = wbuf[off + 4 + j] / 255.0

    return {
        'vertices': positions,
        'uvs': uvs,
        'uvs1': None,
        'uvs3': None,
        'normals': norms,
        'indices': indices,
        'weighted_vertices': 0,
        'bone_indices': bone_indices,
        'bone_weights': bone_weights,
        'skeleton_bones': bones,
        'bone_count': len(bones) if bones else 0,
        'animated': has_skin,
        'version': 0,
        'name': ''
    }

def parse_17(data, filename):
    bone_count = has_skeleton_flag(data)
    has_skin = bone_count != 0
    if "StripAnim" in filename:
        vip = 0x4061; iip = 0x4065; vs = 0x408D
        vnum = _u32(data, vip)
        inum = _u32(data, iip)
        vbuf_len = vnum * 16
        vbuf = data[vs : vs+vbuf_len]
        gap = vbuf_len // 4
        us = vs + vbuf_len + gap
        uvbuf = data[us : us+vbuf_len]
        idx_s = us + vbuf_len + vnum * 8
        ibuf = data[idx_s : idx_s + inum*4]
    else:
        p01 = data.find(b'\x01')
        if p01 == -1: return None
        vip = p01 + 45; iip = 0x75; vs = 0x9D
        vnum = _u32(data, vip)
        inum = _u32(data, iip)
        vbuf_len = vnum * 16
        idx_end = vs + vbuf_len + vbuf_len//4 + vbuf_len + inum*4
        if not (0 < vnum < 500000 and 0 < inum < 3000000 and idx_end <= len(data)):
            if has_skin:
                bones = scan_inline_bones(data, bone_count)
                return {'vertices': [], 'uvs': [], 'indices': [], 'skeleton_bones': bones, 'animated': has_skin}
            return None
        vbuf = data[vs : vs+vbuf_len]
        gap = vbuf_len // 4
        us = vs + vbuf_len + gap
        uvbuf = data[us : us+vbuf_len]
        idx_s = us + vbuf_len
        ibuf = data[idx_s : idx_s + inum*4]

    verts = []
    for i in range(vnum):
        off = i * 16
        verts.extend(struct.unpack_from('<3f', vbuf, off))
    uvs = []
    for i in range(vnum):
        off = i * 16
        uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off)[0]))
        uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off+2)[0]))
    indices = list(struct.unpack(f'<{inum}I', ibuf))
    bones = scan_inline_bones(data, bone_count) if has_skin else []
    return {
        'vertices': verts,
        'uvs': uvs,
        'uvs1': None,
        'uvs3': None,
        'normals': None,
        'indices': indices,
        'weighted_vertices': 0,
        'bone_indices': None,
        'bone_weights': None,
        'skeleton_bones': bones,
        'bone_count': len(bones),
        'animated': has_skin,
        'version': 0x17
    }

def scan_inline_bones(data, bone_count):
    if bone_count <= 0 or bone_count > 4096:
        return []
    marker = data.find(b'RigRef')
    if marker < 0:
        return []
    s = marker
    while s > 0 and 32 <= data[s-1] < 127:
        s -= 1
    return build_bones_from_block(data, bone_count, s)

def parse_1A(data, filename):
    bone_count = has_skeleton_flag(data)
    has_skin = bone_count != 0
    vco = 0x66; ico = 0x6A; vs = 0x92
    vnum = _u32(data, vco)
    inum = _u32(data, ico)
    vbuf_len = vnum * 16
    vbuf = data[vs : vs+vbuf_len]
    gap = vbuf_len // 4
    us = vs + vbuf_len + gap
    uvbuf = data[us : us+vbuf_len]
    idx_s = us + vbuf_len + (vnum * 8 if has_skin else 0)
    ibuf = data[idx_s : idx_s + inum*4]

    verts = []
    for i in range(vnum):
        off = i * 16
        verts.extend(struct.unpack_from('<3f', vbuf, off))
    uvs = []
    for i in range(vnum):
        off = i * 16
        uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off)[0]))
        uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off+2)[0]))
    indices = list(struct.unpack(f'<{inum}I', ibuf))
    bones = scan_inline_bones(data, bone_count) if has_skin else []
    return {
        'vertices': verts,
        'uvs': uvs,
        'uvs1': None,
        'uvs3': None,
        'normals': None,
        'indices': indices,
        'weighted_vertices': 0,
        'bone_indices': None,
        'bone_weights': None,
        'skeleton_bones': bones,
        'bone_count': len(bones),
        'animated': has_skin,
        'version': 0x1A
    }

def parse_1C(data, filename):
    cs = _u32(data, 0x4E)
    us = _u32(data, 0x52)
    dr = lz4_decompress(data[0x56 : 0x56+cs], us)
    tail = data[0x56+cs:]
    has_skin = has_skeleton_flag(data) != 0
    vco = 0x34; ico = 0x38; vs = 0x60
    vnum = _u32(dr, vco)
    inum = _u32(dr, ico)
    vbuf_len = vnum * 16
    vbuf = dr[vs : vs+vbuf_len]
    gap = vbuf_len // 4
    us_start = vs + vbuf_len + gap
    uvbuf = dr[us_start : us_start+vbuf_len]
    idx_s = us_start + vbuf_len + (vnum * 8 if has_skin else 0)
    ibuf = dr[idx_s : idx_s + inum*4]

    verts = []
    for i in range(vnum):
        off = i * 16
        verts.extend(struct.unpack_from('<3f', vbuf, off))
    uvs = []
    for i in range(vnum):
        off = i * 16
        uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off)[0]))
        uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off+2)[0]))
    indices = list(struct.unpack(f'<{inum}I', ibuf))
    bones = parse_tail_bones(tail) if has_skin else []
    return {
        'vertices': verts,
        'uvs': uvs,
        'uvs1': None,
        'uvs3': None,
        'normals': None,
        'indices': indices,
        'weighted_vertices': 0,
        'bone_indices': None,
        'bone_weights': None,
        'skeleton_bones': bones,
        'bone_count': len(bones),
        'animated': has_skin,
        'version': 0x1C
    }

def parse_1E(data, filename):
    cs = _u32(data, 0x4E)
    us = _u32(data, 0x52)
    dr = lz4_decompress(data[0x56 : 0x56+cs], us)
    tail = data[0x56+cs:]
    has_skin = has_skeleton_flag(data) != 0
    vnum = _u32(dr, 0x74)
    inum = _u32(dr, 0x78)
    vs = 0xB3
    vbuf_len = vnum * 16
    vbuf = dr[vs : vs+vbuf_len]
    if has_skin:
        gap = vbuf_len // 4
        us_start = vs + vbuf_len + gap
        uvsz = vbuf_len
        idx_s = us_start + uvsz + vnum * 8
    else:
        gap = vnum * 4 - 4
        us_start = vs + vbuf_len + gap
        uvsz = vnum * 16
        idx_s = us_start + uvsz + 4
    uvbuf = dr[us_start : us_start+uvsz]
    ibuf = dr[idx_s : idx_s + inum*2]

    verts = []
    for i in range(vnum):
        off = i * 16
        verts.extend(struct.unpack_from('<3f', vbuf, off))
    uvs = []
    for i in range(vnum):
        off = i * 16
        uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off+4)[0]))
        uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off+6)[0]))
    indices = list(struct.unpack(f'<{inum}H', ibuf))
    bones = parse_tail_bones(tail) if has_skin else []
    return {
        'vertices': verts,
        'uvs': uvs,
        'uvs1': None,
        'uvs3': None,
        'normals': None,
        'indices': indices,
        'weighted_vertices': 0,
        'bone_indices': None,
        'bone_weights': None,
        'skeleton_bones': bones,
        'bone_count': len(bones),
        'animated': has_skin,
        'version': 0x1E
    }

def parse_1F20(data, version):
    if version == 0x1F:
        hdr = struct.unpack('<18IH3I', data[:86])
        bf = hdr[18]; csz = hdr[20]; usz = hdr[21]; cds = 86
    else:
        hdr = struct.unpack('<18IH4I', data[:90])
        bf = hdr[18]; csz = hdr[21]; usz = hdr[22]; cds = 90
    comp = data[cds : cds+csz]
    dr = lz4_decompress(comp, usz)
    tail = data[cds+csz:]
    has_skin = (bf == 1) or (has_skeleton_flag(data) != 0)
    bones = parse_tail_bones(tail) if has_skin else []
    vnum = _u32(dr, 116)
    inum = _u32(dr, 120)
    vbs = 179
    vbuf = dr[vbs : vbs + vnum*16]
    uvbuf = dr[vbs + vnum*20 : vbs + vnum*36]
    if has_skin:
        wbuf = dr[vbs + vnum*36 : vbs + vnum*44]
        ibuf = dr[vbs + vnum*44 : vbs + vnum*44 + inum*2]
    else:
        wbuf = None
        ibuf = dr[vbs + vnum*36 : vbs + vnum*36 + inum*2]

    verts = []
    for i in range(vnum):
        off = i * 16
        verts.extend(struct.unpack_from('<3f', vbuf, off))
    uvs = []
    for i in range(vnum):
        off = i * 16
        uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off)[0]))
        uvs.append(half_to_float(struct.unpack_from('<H', uvbuf, off+2)[0]))
    indices = list(struct.unpack(f'<{inum}H', ibuf))

    bone_indices = None
    bone_weights = None
    if has_skin and wbuf and bones:
        bone_indices = [0.0] * (vnum * 4)
        bone_weights = [0.0] * (vnum * 4)
        for i in range(vnum):
            off = i * 8
            for j in range(4):
                bone_indices[i*4 + j] = wbuf[off + j]
                bone_weights[i*4 + j] = wbuf[off + 4 + j] / 255.0

    return {
        'vertices': verts,
        'uvs': uvs,
        'uvs1': None,
        'uvs3': None,
        'normals': None,
        'indices': indices,
        'weighted_vertices': vnum if has_skin else 0,
        'bone_indices': bone_indices,
        'bone_weights': bone_weights,
        'skeleton_bones': bones,
        'bone_count': len(bones),
        'animated': has_skin,
        'version': version
    }

def _parse_mesh_any(data: bytes, filename: str = "") -> Dict[str, Any]:
    if len(data) < 4:
        raise ValueError("文件太小")
    magic = data[:4]
    if magic == b'\x17\x00\x00\x00' or magic == b'\x18\x00\x00\x00':
        return parse_17(data, filename)
    elif magic == b'\x19\x00\x00\x00' or magic == b'\x1a\x00\x00\x00' or magic == b'\x1b\x00\x00\x00':
        return parse_1A(data, filename)
    elif magic == b'\x1c\x00\x00\x00' or magic == b'\x1d\x00\x00\x00':
        return parse_1C(data, filename)
    elif magic == b'\x1e\x00\x00\x00':
        return parse_1E(data, filename)
    elif magic == b'\x1f\x00\x00\x00':
        return parse_1F20(data, 0x1F)
    elif magic == b'\x20\x00\x00\x00':
        return parse_1F20(data, 0x20)
    else:
        if 'zippos' in filename.lower():
            payload, tail, has_skin = _zippos_decompress(data)
            bones = parse_tail_bones(tail) if has_skin else []
            result = parse_zippos_payload(payload, has_skin, bones)
            result['version'] = 0
            return result
        raise ValueError(f"不支持的 mesh 版本: {magic.hex()}")

def _zippos_decompress(data):
    magic = data[:4]
    if magic in (b'\x1c\x00\x00\x00', b'\x1d\x00\x00\x00', b'\x1e\x00\x00\x00'):
        cs = _u32(data, 0x4E)
        us = _u32(data, 0x52)
        payload = lz4_decompress(data[0x56 : 0x56+cs], us)
        tail = data[0x56+cs:]
        has_skin = has_skeleton_flag(data) != 0
        return payload, tail, has_skin
    if magic == b'\x1f\x00\x00\x00':
        hdr = struct.unpack('<18IH3I', data[:86])
        bf = hdr[18]; csz = hdr[20]; usz = hdr[21]; cds = 86
    elif magic == b'\x20\x00\x00\x00':
        hdr = struct.unpack('<18IH4I', data[:90])
        bf = hdr[18]; csz = hdr[21]; usz = hdr[22]; cds = 90
    else:
        raise ValueError("ZipPos: unsupported header")
    payload = lz4_decompress(data[cds : cds+csz], usz)
    tail = data[cds+csz:]
    has_skin = (bf == 1) or (has_skeleton_flag(data) != 0)
    return payload, tail, has_skin

def read_mesh(data: bytes, filename: str = "") -> Dict[str, Any]:
    result = _parse_mesh_any(data, filename)
    if result is None:
        raise ValueError("解析失败")
    return {
        'name': result.get('name', os.path.splitext(os.path.basename(filename))[0]),
        'version': result.get('version', 0),
        'animated': result.get('animated', False),
        'vertices': result['vertices'],
        'uvs': result['uvs'],
        'uvs1': result.get('uvs1'),
        'uvs3': result.get('uvs3'),
        'normals': result.get('normals'),
        'indices': result['indices'],
        'weighted_vertices': result.get('weighted_vertices', 0),
        'bone_indices': result.get('bone_indices'),
        'bone_weights': result.get('bone_weights'),
        'skeleton_bones': result.get('skeleton_bones'),
        'bone_count': result.get('bone_count', 0)
    }

# =============================================================================
# 旧版 .meshes 解析器（仅支持新版 GEO0，已验证正确）
# =============================================================================
LVL0_MAGIC = 0x304C564C
MAP_VERTEX_SIZE = 36
MAP_CHUNK_SIZE = 56

def parse_meshes(data: bytes) -> Dict:
    if len(data) < 12:
        raise ValueError("文件过短")
    if struct.unpack('<I', data[:4])[0] != LVL0_MAGIC:
        raise ValueError("非 LVL0 文件")
    version = struct.unpack('<I', data[4:8])[0]
    toc_count = data[8]
    geo0_offset = -1
    p = 12
    for _ in range(toc_count):
        if p + 12 > len(data):
            break
        type_str = read_string(data, p, 4)
        seg_off = struct.unpack('<I', data[p+4:p+8])[0]
        seg_len = struct.unpack('<I', data[p+8:p+12])[0]
        if type_str == 'GEO0':
            geo0_offset = seg_off
        p += 12
    if geo0_offset < 0:
        raise ValueError("未找到 GEO0 段")
    return _parse_geo0(data, geo0_offset, version)

def _parse_geo0(data: bytes, offset: int, version: int) -> Dict:
    pos = offset
    index_count = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
    vertex_count = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
    chunk_count = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
    cloud_count = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
    subchunk_count = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
    compressed_size = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
    if compressed_size <= 0 or pos + compressed_size > len(data):
        raise ValueError("压缩块大小异常")
    compressed = data[pos:pos+compressed_size]
    pos += compressed_size
    raw_verts = meshopt_decode_vertex_buffer(vertex_count, MAP_VERTEX_SIZE, compressed)
    positions = []
    normals = []
    colors = []
    min_coord = [float('inf')]*3
    max_coord = [-float('inf')]*3
    rv = raw_verts
    for i in range(vertex_count):
        base = i * MAP_VERTEX_SIZE
        px = struct.unpack('<f', rv[base:base+4])[0]
        py = struct.unpack('<f', rv[base+4:base+8])[0]
        pz = struct.unpack('<f', rv[base+8:base+12])[0]
        positions.extend([px, py, pz])
        if px < min_coord[0]: min_coord[0]=px
        if py < min_coord[1]: min_coord[1]=py
        if pz < min_coord[2]: min_coord[2]=pz
        if px > max_coord[0]: max_coord[0]=px
        if py > max_coord[1]: max_coord[1]=py
        if pz > max_coord[2]: max_coord[2]=pz
        nx = (rv[base+12] << 24 >> 24) / 127.0
        ny = (rv[base+13] << 24 >> 24) / 127.0
        nz = (rv[base+14] << 24 >> 24) / 127.0
        length = (nx*nx + ny*ny + nz*nz)**0.5 or 1.0
        normals.extend([nx/length, ny/length, nz/length])
        # 材质颜色
        m0 = rv[base + 16]
        m1 = rv[base + 17]
        m2 = rv[base + 18]
        m3 = rv[base + 19]
        w0 = rv[base + 20] / 255.0
        w1 = rv[base + 21] / 255.0
        w2 = rv[base + 22] / 255.0
        w3 = rv[base + 23] / 255.0
        tw = w0 + w1 + w2 + w3
        # 颜色映射表（与之前一致）
        color_map = {
            0: [0.5, 0.5, 0.5], 2: [0.6, 0.7, 0.8], 3: [0.2, 0.2, 0.2],
            4: [0.9, 0.8, 0.6], 5: [0.6, 0.45, 0.3], 6: [0.3, 0.3, 0.3],
            7: [0.65, 0.5, 0.35], 16: [0.55, 0.5, 0.45], 17: [0.6, 0.5, 0.35],
            18: [0.65, 0.6, 0.5], 19: [0.4, 0.35, 0.3], 20: [0.5, 0.5, 0.55],
            21: [0.9, 0.8, 0.3], 22: [0.7, 0.85, 0.95], 23: [0.8, 0.8, 0.85],
            24: [0.75, 0.75, 0.8], 25: [0.7, 0.7, 0.75], 26: [0.6, 0.45, 0.35],
            27: [0.4, 0.35, 0.25], 28: [0.35, 0.35, 0.35], 29: [0.85, 0.85, 0.8],
            30: [0.6, 0.45, 0.3], 31: [0.8, 0.7, 0.6], 32: [0.85, 0.78, 0.55],
            33: [0.7, 0.65, 0.45], 34: [0.9, 0.85, 0.65], 35: [0.95, 0.95, 0.98],
            36: [0.75, 0.68, 0.45], 37: [0.45, 0.4, 0.3], 48: [0.4, 0.6, 0.3],
            49: [0.3, 0.5, 0.25], 50: [0.5, 0.7, 0.35], 51: [0.35, 0.55, 0.3],
            52: [0.7, 0.5, 0.5], 80: [0.9, 0.9, 1.0]
        }
        def get_color(idx):
            return color_map.get(idx, [0.7, 0.7, 0.7])
        if tw < 0.001:
            c = get_color(m0)
            colors.extend(c)
        else:
            c0 = get_color(m0); c1 = get_color(m1); c2 = get_color(m2); c3 = get_color(m3)
            r = (c0[0] * w0 + c1[0] * w1 + c2[0] * w2 + c3[0] * w3) / tw
            g = (c0[1] * w0 + c1[1] * w1 + c2[1] * w2 + c3[1] * w3) / tw
            b = (c0[2] * w0 + c1[2] * w1 + c2[2] * w2 + c3[2] * w3) / tw
            colors.extend([r, g, b])
    
    local_indices = None
    if index_count > 0 and pos + index_count <= len(data):
        local_indices = data[pos:pos+index_count]
        pos += index_count
    chunks = []
    total_chunks = chunk_count + cloud_count
    for _ in range(total_chunks):
        if pos + MAP_CHUNK_SIZE > len(data):
            break
        idx_start = struct.unpack('<I', data[pos:pos+4])[0]
        vtx_start = struct.unpack('<I', data[pos+4:pos+8])[0]
        idx_cnt = struct.unpack('<H', data[pos+12:pos+14])[0]
        chunks.append((idx_start, vtx_start, idx_cnt))
        pos += MAP_CHUNK_SIZE
    global_indices = []
    cloud_indices = []
    if local_indices:
        for ci in range(min(chunk_count, len(chunks))):
            idx_start, vtx_start, idx_cnt = chunks[ci]
            for j in range(idx_cnt):
                li = idx_start + j
                if li >= index_count:
                    break
                gi = local_indices[li] + vtx_start
                if 0 <= gi < vertex_count:
                    global_indices.append(gi)
        for ci in range(chunk_count, min(chunk_count + cloud_count, len(chunks))):
            idx_start, vtx_start, idx_cnt = chunks[ci]
            for j in range(idx_cnt):
                li = idx_start + j
                if li >= index_count:
                    break
                gi = local_indices[li] + vtx_start
                if 0 <= gi < vertex_count:
                    cloud_indices.append(gi)
    return {'positions': positions, 'normals': normals, 'colors': colors,
            'indices': global_indices, 'cloud_indices': cloud_indices,
            'vertex_count': vertex_count, 'index_count': len(global_indices),
            'chunk_count': chunk_count, 'cloud_count': cloud_count,
            'bounds_min': min_coord, 'bounds_max': max_coord, 'version': version}

# =============================================================================
# 旧版 .meshes 解析器（基于 blender_import_sky_meshes.py，用于 <57）
# =============================================================================

# 材质颜色表 (material_id -> RGB 0-255)
MATERIAL_COLORS = {
    16: (188, 168, 148),     # 沙漠砂岩
    18: (88, 78, 68),        # 洞穴/阴影
    32: (202, 182, 156),     # 浅色岩石
    24: (108, 98, 88),       # 暗色岩石
    48: (238, 208, 148),     # 沙子/沙滩
    49: (105, 168, 82),      # 草地
    12337: (172, 162, 150),  # 岩石A
    12592: (158, 148, 138),  # 岩石B
}

def _material_color(mid):
    if mid in MATERIAL_COLORS:
        return MATERIAL_COLORS[mid]
    h = (mid * 0x9E3779B1) & 0xFFFFFFFF
    r = 80 + (h & 0xFF) % 130
    g = 80 + ((h >> 8) & 0xFF) % 130
    b = 80 + ((h >> 16) & 0xFF) % 130
    return (r, g, b)

class BinaryStreamReader:
    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_offset = 0

    def align(self):
        if self.bit_offset != 0:
            self.bit_offset = 0
            self.byte_pos += 1

    def remaining(self) -> int:
        return len(self.data) - self.byte_pos

    def _check(self, n: int):
        if self.byte_pos + n > len(self.data):
            raise EOFError(f"read at {self.byte_pos}+{n} > len={len(self.data)}")

    def read_bytes(self, n: int) -> bytes:
        self.align()
        if n <= 0:
            return b""
        self._check(n)
        res = self.data[self.byte_pos:self.byte_pos + n]
        self.byte_pos += n
        return res

    def read_uint32(self) -> int:
        return struct.unpack('<I', self.read_bytes(4))[0]

    def read_int32(self) -> int:
        return struct.unpack('<i', self.read_bytes(4))[0]

    def read_float(self) -> float:
        return struct.unpack('<f', self.read_bytes(4))[0]

    def read_float3(self) -> list:
        return list(struct.unpack('<3f', self.read_bytes(12)))

    def read_bits(self, count: int) -> int:
        value = 0
        bits_read = 0
        while bits_read < count:
            if self.byte_pos >= len(self.data):
                raise EOFError("read_bits past end")
            byte = self.data[self.byte_pos]
            remaining = 8 - self.bit_offset
            take = min(count - bits_read, remaining)
            mask = (1 << take) - 1
            chunk = (byte >> self.bit_offset) & mask
            value |= chunk << bits_read
            bits_read += take
            self.bit_offset += take
            if self.bit_offset >= 8:
                self.bit_offset = 0
                self.byte_pos += 1
        return value

def parse_toc(data: bytes) -> list:
    toc_start = 0x08
    entry_count = data[toc_start]
    entries = []
    for i in range(entry_count):
        base = toc_start + 4 + i * 12
        name = data[base:base + 4].rstrip(b'\x00').decode('ascii', errors='replace')
        offset = struct.unpack_from('<I', data, base + 4)[0]
        length = struct.unpack_from('<I', data, base + 8)[0]
        entries.append((name, offset, length))
    return entries

def skip_generic_meshes(stream: BinaryStreamReader):
    total = stream.read_uint32()
    if total == 0 or total > 100000:
        return

    for i in range(total):
        name_len = stream.read_uint32()
        if name_len > 1024:
            stream.byte_pos -= 4
            break
        stream.read_bytes(name_len)
        stream.read_uint32()
        submesh_count = stream.read_uint32()
        sep = stream.read_bytes(1)[0]
        if sep == 0x00:
            mode = 0
            cc = stream.read_uint32()
            gg = stream.read_uint32()
            stream.read_bytes(12 * cc + gg)
        elif sep == 0x01:
            mode = 1
            stream.read_bytes(4)
            gg = stream.read_uint32()
            stream.read_bytes(12 + gg)
        else:
            mode = 0
            cc = stream.read_uint32()
            gg = stream.read_uint32()
            stream.read_bytes(12 * cc + gg)

        for j in range(1, submesh_count):
            if mode == 0:
                cc = stream.read_uint32()
                gg = stream.read_uint32()
                stream.read_bytes(12 * cc + gg)
            elif mode == 1:
                stream.read_bytes(4)
                gg = stream.read_uint32()
                stream.read_bytes(12 + gg)

def decode_terrain_patches(stream: BinaryStreamReader, color_mode: str = 'MATERIAL') -> dict:
    result = {'positions': [], 'colors': [], 'indices': []}
    patch_count = stream.read_uint32()
    if patch_count == 0 or patch_count > 100000:
        return result

    global_verts = []
    global_colors = []
    global_indices = []

    for pi in range(patch_count):
        ext_id = stream.read_int32()
        stream.read_bits(1)
        stream.read_bits(1)
        stream.align()
        stream.read_float3()
        stream.read_float3()
        vert_count = stream.read_uint32()
        index_count = stream.read_uint32()

        if vert_count > 500000 or index_count > 1500000:
            break

        vert_bytes = stream.read_bytes(36 * vert_count)
        px, py, pz = [0.0]*vert_count, [0.0]*vert_count, [0.0]*vert_count
        cr, cg, cb, ca = [0]*vert_count, [0]*vert_count, [0]*vert_count, [0]*vert_count
        mat_ids = [0]*vert_count

        for v in range(vert_count):
            off = v * 36
            px[v], py[v], pz[v] = struct.unpack_from('<3f', vert_bytes, off)
            mat_ids[v] = struct.unpack_from('<I', vert_bytes, off + 16)[0]
            cr[v], cg[v], cb[v], ca[v] = vert_bytes[off + 24:off + 28]

        some_count = stream.read_uint32()
        if some_count <= stream.remaining():
            stream.read_bytes(some_count)

        stream.read_float3(); stream.read_float3()
        stream.read_float()
        for _ in range(3):
            stream.read_uint32()

        count_A = stream.read_uint32()
        if count_A > 50000:
            break

        for _ in range(count_A):
            stream.read_uint32(); stream.read_uint32()

        count_A_repeat = stream.read_uint32()
        count_B = stream.read_uint32()
        count_C = stream.read_uint32()

        if count_B > stream.remaining() // 4 or count_C > stream.remaining() // 2:
            break

        if count_B > 0:
            stream.read_bytes(count_B * 4)
        if count_C > 0:
            stream.read_bytes(count_C * 2)
        for _ in range(count_A_repeat):
            stream.read_uint32()

        triangles = []
        if count_B == 0 and index_count > 0:
            if index_count <= stream.remaining() // 2:
                raw = stream.read_bytes(index_count * 2)
                raw_indices = list(struct.unpack(f'<{index_count}H', raw))
                for t in range(0, len(raw_indices) - 2, 3):
                    i1, i2, i3 = raw_indices[t], raw_indices[t+1], raw_indices[t+2]
                    if i1 == i2 or i2 == i3 or i1 == i3:
                        continue
                    if max(i1, i2, i3) >= vert_count:
                        continue
                    triangles.append((i1, i2, i3))

        base_idx = len(global_verts)
        for i in range(vert_count):
            global_verts.append((px[i], py[i], pz[i]))
            if color_mode == 'MATERIAL':
                r, g, b = _material_color(mat_ids[i])
                global_colors.append((r, g, b))
            else:
                global_colors.append((cr[i], cg[i], cb[i]))
        for (a, b, c) in triangles:
            global_indices.append((base_idx + a, base_idx + b, base_idx + c))

    result['positions'] = global_verts
    result['colors'] = global_colors
    result['indices'] = global_indices
    return result

def decode_cloud_v6(stream: BinaryStreamReader) -> dict:
    result = {'positions': [], 'colors': [], 'indices': []}
    stream.align()
    save_pos = stream.byte_pos
    cloud_flag = stream.read_uint32()
    if cloud_flag == 0 or cloud_flag == 0xFFFFFFFF:
        return result

    try:
        hdr = [stream.read_uint32() for _ in range(6)]
        dim_x, dim_y, dim_z = hdr[3], hdr[4], hdr[5]
        if dim_x == 0 or dim_y == 0 or dim_z == 0 or dim_x > 0x10000:
            stream.byte_pos = save_pos
            return result

        sparse_size = dim_x * dim_y * dim_z
        if sparse_size > stream.remaining():
            stream.byte_pos = save_pos
            return result

        stream.read_bytes(sparse_size)
        if stream.remaining() < 4:
            return result
        count1 = stream.read_uint32()
        if count1 > 0 and count1 * 6 <= stream.remaining():
            stream.read_bytes(count1 * 6)

        if stream.remaining() < 12:
            return result
        v3 = [stream.read_uint32() for _ in range(3)]
        for sz in v3:
            if 0 < sz <= stream.remaining():
                stream.read_bytes(sz)

        if stream.remaining() >= 8:
            stream.read_float(); stream.read_uint32()

        if stream.remaining() >= 12:
            v6 = [stream.read_uint32() for _ in range(3)]
            vert_count = v6[1]
            idx_count = v6[2]
            if 0 < vert_count <= stream.remaining() // 16:
                v6r = stream.read_bytes(16 * vert_count)
                px, py, pz = [0.0]*vert_count, [0.0]*vert_count, [0.0]*vert_count
                for v in range(vert_count):
                    off = v * 16
                    px[v], py[v], pz[v] = struct.unpack_from('<3f', v6r, off)
                for i in range(vert_count):
                    result['positions'].append((px[i], py[i], pz[i]))
                result['colors'].extend([(255, 255, 255)] * vert_count)

            if 0 < idx_count <= stream.remaining() // 2:
                v6i = stream.read_bytes(2 * idx_count)
                indices = list(struct.unpack(f'<{idx_count}H', v6i))
                triangles = []
                for t in range(0, len(indices) - 2, 3):
                    i1, i2, i3 = indices[t], indices[t+1], indices[t+2]
                    if i1 == i2 or i2 == i3 or i1 == i3:
                        continue
                    if max(i1, i2, i3) >= vert_count:
                        continue
                    triangles.append((i1, i2, i3))
                base = len(result['positions']) - vert_count
                for (a, b, c) in triangles:
                    result['indices'].append((base + a, base + b, base + c))

        return result

    except EOFError:
        stream.byte_pos = save_pos
        return {'positions': [], 'colors': [], 'indices': []}

def decode_cloud_mesh(stream: BinaryStreamReader) -> tuple:
    groups = []
    mesh16 = {'vertices': [], 'colors': [], 'indices': []}

    if stream.remaining() < 5:
        return groups, mesh16

    pos_before = stream.byte_pos
    try:
        flag = stream.read_bytes(1)[0]
        if flag != 0x01:
            stream.byte_pos = pos_before
            return groups, mesh16

        group_count = stream.read_uint32()
        if group_count == 0 or group_count > 50000:
            stream.byte_pos = pos_before
            return groups, mesh16

        for gi in range(group_count):
            if stream.remaining() < 8:
                break
            vert_count = stream.read_uint32()
            if vert_count == 0 or vert_count > stream.remaining() // 40:
                stream.byte_pos -= 4
                break

            vert_raw = stream.read_bytes(40 * vert_count)
            vc_g = vert_count
            gpx, gpy, gpz = [0.0]*vc_g, [0.0]*vc_g, [0.0]*vc_g
            gcr, gcg, gcb, gca = [0]*vc_g, [0]*vc_g, [0]*vc_g, [0]*vc_g
            gmat = [0]*vc_g

            for v in range(vc_g):
                off = v * 40
                gpx[v], gpy[v], gpz[v] = struct.unpack_from('<3f', vert_raw, off)
                gcr[v], gcg[v], gcb[v], gca[v] = vert_raw[off + 24:off + 28]
                gmat[v] = struct.unpack_from('<I', vert_raw, off + 16)[0]

            if stream.remaining() < 4:
                break
            idx_count = stream.read_uint32()
            if idx_count > stream.remaining() // 2:
                break
            idx_raw = stream.read_bytes(2 * idx_count)
            indices = list(struct.unpack(f'<{idx_count}H', idx_raw))

            triangles = []
            for t in range(0, len(indices) - 2, 3):
                i1, i2, i3 = indices[t], indices[t+1], indices[t+2]
                if i1 == i2 or i2 == i3 or i1 == i3:
                    continue
                if max(i1, i2, i3) >= vert_count:
                    continue
                triangles.append((i1, i2, i3))

            verts = [(gpx[i], gpy[i], gpz[i]) for i in range(vc_g)]
            colors = [(255, 255, 255)] * vc_g
            groups.append({'vertices': verts, 'colors': colors, 'indices': triangles})

        if stream.remaining() >= 12:
            N = stream.read_uint32()
            if 0 < N <= 100 and stream.remaining() >= 8:
                a = stream.read_uint32()
                b = stream.read_uint32()
                if 0 < a <= stream.remaining() // 16 and 0 < b <= stream.remaining() // 2:
                    vert_r16 = stream.read_bytes(16 * a)
                    p16x, p16y, p16z = [0.0]*a, [0.0]*a, [0.0]*a
                    for v in range(a):
                        off = v * 16
                        p16x[v], p16y[v], p16z[v] = struct.unpack_from('<3f', vert_r16, off)
                    idx_r16 = stream.read_bytes(2 * b)
                    indices_16 = list(struct.unpack(f'<{b}H', idx_r16))

                    triangles_16 = []
                    for t in range(0, len(indices_16) - 2, 3):
                        i1, i2, i3 = indices_16[t], indices_16[t+1], indices_16[t+2]
                        if i1 == i2 or i2 == i3 or i1 == i3:
                            continue
                        if max(i1, i2, i3) >= a:
                            continue
                        triangles_16.append((i1, i2, i3))

                    verts16 = [(p16x[i], p16y[i], p16z[i]) for i in range(a)]
                    colors16 = [(255, 255, 255)] * a
                    mesh16 = {'vertices': verts16, 'colors': colors16, 'indices': triangles_16}

        return groups, mesh16

    except EOFError:
        stream.byte_pos = pos_before
        return [], {'vertices': [], 'colors': [], 'indices': []}

def _parse_meshes_unified(data: bytes, color_mode: str = 'MATERIAL') -> Dict:
    if data[:4] == b'LVL0':
        entries = parse_toc(data)
        if not entries:
            raise ValueError("LVL0 容器无 LOD 数据")
        lod_entry = None
        for name, off, length in entries:
            if 'LOD' in name:
                lod_entry = (off, length)
                break
        if not lod_entry:
            raise ValueError("未找到 LOD 段")
        offset, length = lod_entry
        compressed = data[offset:offset + length]
        try:
            data = lz4.block.decompress(compressed, uncompressed_size=0xC00000)
        except Exception as e:
            raise RuntimeError(f"LZ4 解压失败: {e}")

    stream = BinaryStreamReader(data)
    skip_generic_meshes(stream)

    terrain_result = decode_terrain_patches(stream, color_mode)
    cloud_v6 = decode_cloud_v6(stream)
    cloud_groups, mesh16 = decode_cloud_mesh(stream)

    all_positions = list(terrain_result['positions'])
    all_colors = list(terrain_result['colors'])
    all_indices = list(terrain_result['indices'])
    cloud_indices = []

    if cloud_v6.get('positions'):
        cv_pos = cloud_v6['positions']
        cv_col = cloud_v6['colors']
        cv_idx = cloud_v6['indices']
        if cv_pos:
            base = len(all_positions)
            all_positions.extend(cv_pos)
            all_colors.extend(cv_col)
            for (a, b, c) in cv_idx:
                cloud_indices.append((base + a, base + b, base + c))

    for g in cloud_groups:
        if g['vertices']:
            base = len(all_positions)
            all_positions.extend(g['vertices'])
            all_colors.extend(g['colors'])
            for (a, b, c) in g['indices']:
                cloud_indices.append((base + a, base + b, base + c))

    if mesh16 and mesh16['vertices']:
        base = len(all_positions)
        all_positions.extend(mesh16['vertices'])
        all_colors.extend(mesh16['colors'])
        for (a, b, c) in mesh16['indices']:
            cloud_indices.append((base + a, base + b, base + c))

    flat_indices = []
    for (a, b, c) in all_indices:
        flat_indices.extend([a, b, c])
    flat_cloud_indices = []
    for (a, b, c) in cloud_indices:
        flat_cloud_indices.extend([a, b, c])

    flat_positions = []
    for p in all_positions:
        flat_positions.extend(p)
    flat_colors = []
    for c in all_colors:
        flat_colors.extend([c[0]/255.0, c[1]/255.0, c[2]/255.0])

    if flat_positions:
        arr = np.array(flat_positions).reshape(-1, 3)
        min_coord = arr.min(axis=0).tolist()
        max_coord = arr.max(axis=0).tolist()
    else:
        min_coord = [0, 0, 0]
        max_coord = [0, 0, 0]

    return {
        'positions': flat_positions,
        'normals': [],
        'colors': flat_colors,
        'indices': flat_indices,
        'cloud_indices': flat_cloud_indices,
        'vertex_count': len(all_positions),
        'index_count': len(flat_indices)//3,
        'cloud_count': len(flat_cloud_indices)//3,
        'bounds_min': min_coord,
        'bounds_max': max_coord,
        'version': 0
    }

# =============================================================================
# 查找 .mesh 文件
# =============================================================================
def find_mesh_file(mesh_root: str, res_name: str) -> Optional[Path]:
    base = Path(mesh_root)
    res_lower = res_name.lower()
    for f in base.rglob('*.mesh'):
        stem = f.stem.lower()
        if stem == res_lower or stem.startswith(res_lower + '_'):
            return f
    return None

# =============================================================================
# 导出 GLB / OBJ
# =============================================================================
def export_to_glb(meshes: List[Dict], output_path: Path):
    scene = trimesh.Scene()
    for mesh in meshes:
        verts = np.array(mesh['vertices']).reshape(-1, 3)
        idx = np.array(mesh['indices'], dtype=np.int64)
        mat = np.array(mesh['matrix']).reshape(4, 4)
        if not np.allclose(mat, np.eye(4)):
            verts = trimesh.transform_points(verts, mat)
        if 'colors' in mesh and mesh['colors'] is not None:
            colors = np.array(mesh['colors']).reshape(-1, 3)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)
            colors = np.concatenate([colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)], axis=1)
            vertex_colors = colors
        else:
            vertex_colors = None
        tri = trimesh.Trimesh(vertices=verts, faces=idx.reshape(-1, 3), vertex_colors=vertex_colors)
        if 'normals' in mesh and mesh['normals'] is not None:
            tri.vertex_normals = np.array(mesh['normals']).reshape(-1, 3)
        scene.add_geometry(tri, node_name=mesh.get('name', 'mesh'))
    scene.export(str(output_path), file_type='glb')

def export_to_obj(meshes: List[Dict], output_path: Path):
    scene = trimesh.Scene()
    for mesh in meshes:
        verts = np.array(mesh['vertices']).reshape(-1, 3)
        idx = np.array(mesh['indices'], dtype=np.int64)
        mat = np.array(mesh['matrix']).reshape(4, 4)
        if not np.allclose(mat, np.eye(4)):
            verts = trimesh.transform_points(verts, mat)
        if 'colors' in mesh and mesh['colors'] is not None:
            colors = np.array(mesh['colors']).reshape(-1, 3)
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)
            colors = np.concatenate([colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)], axis=1)
            vertex_colors = colors
        else:
            vertex_colors = None
        tri = trimesh.Trimesh(vertices=verts, faces=idx.reshape(-1, 3), vertex_colors=vertex_colors)
        if 'normals' in mesh and mesh['normals'] is not None:
            tri.vertex_normals = np.array(mesh['normals']).reshape(-1, 3)
        scene.add_geometry(tri, node_name=mesh.get('name', 'mesh'))
    scene.export(str(output_path), file_type='obj')

# =============================================================================
# 核心转换函数
# =============================================================================
def convert_map(map_folder: str, mesh_root: str, output_root: str, fmt: str = 'glb',
                export_markers: bool = True, color_mode: str = 'MATERIAL'):
    map_path = Path(map_folder)
    if not map_path.is_dir():
        print(f"错误：{map_folder} 不是目录")
        return
    map_name = map_path.name
    out_dir = Path(output_root) / map_name
    out_dir.mkdir(parents=True, exist_ok=True)

    meshes_files = list(map_path.glob('*.meshes'))
    if not meshes_files:
        print(f"警告：{map_folder} 中没有 .meshes 文件")
        return
    meshes_file = meshes_files[0]
    level_name = meshes_file.stem
    print(f"\n=== 处理地图: {map_name} (文件: {level_name}) ===")

    bin_file = map_path / 'Objects.level.bin'
    if not bin_file.exists():
        print("警告：缺少 Objects.level.bin，将跳过物件")
        json_data = None
    else:
        print(f"Objects.level.bin 大小: {bin_file.stat().st_size} 字节")
        try:
            json_data = bin_to_json(str(bin_file), translate=False)
            print("Objects.level.bin 解析成功")
        except Exception as e:
            print(f"bin_to_json 解析失败: {e}")
            json_data = None

    # 解析地形（根据版本选择解析器）
    try:
        data = meshes_file.read_bytes()
        if len(data) < 8:
            raise ValueError("文件过短")
        file_version = struct.unpack('<I', data[4:8])[0]
        print(f"文件版本: {file_version}")

        if file_version >= 57:
            # 新版使用原版 parse_meshes（已验证正确）
            terrain_data = parse_meshes(data)
            print("使用新版解析器 (GEO0)")
        else:
            # 旧版使用统一解析器（支持 v54 以下及 v55~56）
            terrain_data = _parse_meshes_unified(data, color_mode)
            print("使用旧版统一解析器")

        print(f"地形: 顶点 {terrain_data['vertex_count']}, 面 {len(terrain_data['indices'])//3}, 云面 {len(terrain_data['cloud_indices'])//3}")
    except Exception as e:
        print(f"地形解析失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 提取物件实例和标记点
    instances = []
    markers = []
    if json_data:
        try:
            instances = extract_level_meshes_from_json(json_data)
            if export_markers:
                markers = extract_level_markers_from_json(json_data)
            print(f"物件实例数: {len(instances)}")
            if export_markers:
                total_markers = sum(len(g['points']) for g in markers)
                print(f"标记点总数: {total_markers}")
        except Exception as e:
            print(f"提取物件/标记点失败: {e}")

    # 构建网格列表
    meshes_to_export = []
    # 1. 地形
    pos = np.array(terrain_data['positions']).reshape(-1, 3) if terrain_data['positions'] else np.empty((0,3))
    norm = np.array(terrain_data['normals']).reshape(-1, 3) if terrain_data['normals'] else None
    col = np.array(terrain_data['colors']).reshape(-1, 3) if terrain_data['colors'] else None
    idx = np.array(terrain_data['indices'], dtype=np.uint32) if terrain_data['indices'] else np.empty((0,), dtype=np.uint32)
    mesh_terrain = {
        'name': f"{level_name}_terrain",
        'vertices': pos,
        'normals': norm,
        'colors': col,
        'indices': idx,
        'matrix': np.eye(4, dtype=np.float32)
    }
    meshes_to_export.append(mesh_terrain)
    # 2. 云
    if terrain_data['cloud_indices']:
        cloud_idx = np.array(terrain_data['cloud_indices'], dtype=np.uint32)
        mesh_cloud = {
            'name': f"{level_name}_cloud",
            'vertices': pos,
            'normals': norm,
            'colors': col,
            'indices': cloud_idx,
            'matrix': np.eye(4, dtype=np.float32)
        }
        meshes_to_export.append(mesh_cloud)

    # 3. 物件实例（需缓存 mesh 数据）
    mesh_cache = {}
    loaded_count = 0
    for inst in instances:
        res_name = inst['resourceName']
        if res_name not in mesh_cache:
            mesh_file = find_mesh_file(mesh_root, res_name)
            if mesh_file is None:
                print(f"警告：找不到 mesh 文件 {res_name}")
                continue
            try:
                mesh_data = read_mesh(mesh_file.read_bytes(), mesh_file.name)
                if not mesh_data['vertices']:
                    print(f"警告：mesh {res_name} 无顶点数据")
                    continue
                verts = np.array(mesh_data['vertices']).reshape(-1, 3)
                norms = np.array(mesh_data['normals']).reshape(-1, 3) if mesh_data['normals'] else None
                colors = np.full((verts.shape[0], 3), 0.7, dtype=np.float32)
                indices = np.array(mesh_data['indices'], dtype=np.uint32)
                mesh_cache[res_name] = (verts, norms, colors, indices)
            except Exception as e:
                print(f"解析 mesh {res_name} 失败: {e}")
                continue
        verts, norms, colors, indices = mesh_cache[res_name]
        mat = np.array(inst['matrix']).reshape(4, 4)
        mesh_obj = {
            'name': f"{level_name}_{res_name}",
            'vertices': verts,
            'normals': norms,
            'colors': colors,
            'indices': indices,
            'matrix': mat
        }
        meshes_to_export.append(mesh_obj)
        loaded_count += 1
    print(f"成功加载物件网格数: {loaded_count}")

    # ===== 4. 标记点 → 小圆球 =====
    if export_markers and markers:
        import math
        MARKER_RADIUS = 0.5
        SEGMENTS = 8
        rings = SEGMENTS // 2
        sphere_verts = []
        sphere_faces = []
        sphere_verts.append((0, MARKER_RADIUS, 0))
        sphere_verts.append((0, -MARKER_RADIUS, 0))
        for i in range(1, rings):
            phi = math.pi * i / rings
            y = MARKER_RADIUS * math.cos(phi)
            r = MARKER_RADIUS * math.sin(phi)
            for j in range(SEGMENTS):
                theta = 2 * math.pi * j / SEGMENTS
                x = r * math.cos(theta)
                z = r * math.sin(theta)
                sphere_verts.append((x, y, z))
        for j in range(SEGMENTS):
            sphere_faces.append((0, 2 + j, 2 + (j + 1) % SEGMENTS))
        base_idx = 2 + (rings - 2) * SEGMENTS
        for j in range(SEGMENTS):
            sphere_faces.append((1, base_idx + (j + 1) % SEGMENTS, base_idx + j))
        for i in range(rings - 2):
            for j in range(SEGMENTS):
                a = 2 + i * SEGMENTS + j
                b = 2 + i * SEGMENTS + (j + 1) % SEGMENTS
                c = 2 + (i + 1) * SEGMENTS + j
                d = 2 + (i + 1) * SEGMENTS + (j + 1) % SEGMENTS
                sphere_faces.append((a, b, d))
                sphere_faces.append((a, d, c))
        sphere_verts = np.array(sphere_verts, dtype=np.float32)
        sphere_faces = np.array(sphere_faces, dtype=np.int64).flatten()

        for group in markers:
            color_hex = group['color'].lstrip('#')
            r = int(color_hex[0:2], 16) / 255.0
            g = int(color_hex[2:4], 16) / 255.0
            b = int(color_hex[4:6], 16) / 255.0
            color_rgb = np.array([r, g, b])
            for pt in group['points']:
                pos = np.array(pt['pos'])
                verts = sphere_verts + pos
                colors = np.tile(color_rgb, (len(verts), 1))
                mesh_obj = {
                    'name': f"marker_{group['key']}_{pt['label']}",
                    'vertices': verts,
                    'normals': None,
                    'colors': colors,
                    'indices': sphere_faces.copy(),
                    'matrix': np.eye(4, dtype=np.float32)
                }
                meshes_to_export.append(mesh_obj)
        print(f"添加了 {sum(len(g['points']) for g in markers)} 个标记点球体")

    if not meshes_to_export:
        print(f"警告：{map_name} 无任何几何数据")
        return

    # 导出模型
    ext = '.glb' if fmt == 'glb' else '.obj'
    out_file = out_dir / f"{level_name}{ext}"
    print(f"导出模型到: {out_file}")
    if fmt == 'glb':
        export_to_glb(meshes_to_export, out_file)
    else:
        export_to_obj(meshes_to_export, out_file)
    print(f"模型导出完成，包含 {len(meshes_to_export)} 个子网格")

    # 导出标记点 JSON
    if export_markers and markers:
        markers_json = []
        for group in markers:
            for pt in group['points']:
                markers_json.append({
                    'type': group['key'],
                    'label': pt['label'],
                    'position': pt['pos'],
                    'color': group['color']
                })
        json_path = out_dir / f"{level_name}_markers.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(markers_json, f, indent=2, ensure_ascii=False)
        print(f"标记点 JSON 已导出: {json_path} ({len(markers_json)} 个点)")
    else:
        if export_markers:
            print("无标记点可导出")

# =============================================================================
# 主交互
# =============================================================================
def main():
    print("=== Sky 地图转模型工具（全融合版，新解析内核） ===")
    print("支持所有 .mesh 版本（0x17~0x20，含 ZipPos）及地图 .meshes（全版本）")
    print()
    mode = input("选择转换模式 [1] 单个地图 [2] 批量: ").strip()
    if mode not in ('1', '2'):
        print("无效选择")
        return
    if mode == '1':
        map_folder = input("请输入地图文件夹路径（包含 .meshes 和 Objects.level.bin）: ").strip()
        map_folders = [(Path(map_folder).name, map_folder)]
    else:
        parent_dir = input("请输入包含多个地图子文件夹的父目录: ").strip()
        if not os.path.isdir(parent_dir):
            print("目录不存在")
            return
        map_folders = []
        for d in os.listdir(parent_dir):
            sub = os.path.join(parent_dir, d)
            if os.path.isdir(sub) and any(f.endswith('.meshes') for f in os.listdir(sub)):
                map_folders.append((d, sub))
        if not map_folders:
            print("未找到任何包含 .meshes 的子文件夹")
            return
        print(f"找到 {len(map_folders)} 个地图文件夹")
        names = [f[0] for f in map_folders]
        if len(names) != len(set(names)):
            print("错误：存在同名地图文件夹，请重命名后重试。")
            return

    mesh_root = input("请输入 mesh 目录（存放所有 .mesh 文件的根目录）: ").strip()
    if not os.path.isdir(mesh_root):
        print("mesh 目录不存在")
        return
    output_root = input("请输入输出根目录（将在其中按地图名创建子目录）: ").strip()
    os.makedirs(output_root, exist_ok=True)

    fmt = input("导出格式 [glb/obj] (默认 glb): ").strip().lower()
    if fmt not in ('glb', 'obj'):
        fmt = 'glb'

    color_mode = input("颜色模式 [material/baked] (默认 material): ").strip().lower()
    if color_mode not in ('material', 'baked'):
        color_mode = 'material'
    color_mode = 'MATERIAL' if color_mode == 'material' else 'BAKED'

    export_markers = input("是否导出标记点？ (y/n, 默认 y): ").strip().lower() != 'n'

    for map_name, folder in map_folders:
        try:
            convert_map(folder, mesh_root, output_root, fmt, export_markers, color_mode)
        except Exception as e:
            print(f"转换 {folder} 失败: {e}")
            import traceback
            traceback.print_exc()
    print("\nok")

if __name__ == '__main__':
    main()