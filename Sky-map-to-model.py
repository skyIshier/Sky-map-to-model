#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sky 地图转模型工具（全融合版）
内置完整 TGCL 解析（移植自 bintojson.py），无需外部文件
支持所有 .mesh 版本（0x17~0x20）及地图 .meshes
用法：直接运行，按终端提示操作。
依赖：pip install lz4 trimesh numpy

作者:十二
作者联系方式:
q:3787533101
邮箱:3787533101@qq.com
项目链接:https://github.com/skyIshier/Sky-map-to-model
"""

import os
import sys
import json
import struct
import re
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import lz4.block
import numpy as np
import trimesh

# =============================================================================
# 第一部分：TGCL 解析器（完整移植自 bintojson.py）
# =============================================================================

# ---------- 翻译表（保留原样） ----------
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

    # 建立类名到字段元数据的映射（用于判断字段类型）
    class_meta = {}
    for cls_name, meta in classes.items():
        if meta is None:
            class_meta[cls_name] = {}
        else:
            class_meta[cls_name] = meta

    for node_name, node_data in nodes.items():
        if not isinstance(node_data, dict):
            continue
        # 节点数据可能包含多个类层次，找到第一个包含 resourceName 和 transform 的类
        for cls_name, fields in node_data.items():
            if not isinstance(fields, dict):
                continue
            # 检查是否有 resourceName 字段（字符串类型）
            res_name = None
            for key in ['resourceName', 'mesh', 'meshName']:
                if key in fields and isinstance(fields[key], str):
                    res_name = fields[key]
                    break
            if not res_name:
                continue
            # 检查是否有 transform 字段（pod 类型，64字节）
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
            # 去重
            key = res_name + '|' + ','.join([str(x) for x in tf])
            if key in seen:
                continue
            seen.add(key)
            # 提取其他属性（shaderParams 等）
            shader_name = fields.get('shaderName', '') if isinstance(fields.get('shaderName'), str) else ''
            diffuse_tex = ''
            norm_tex = ''
            diffuse2_tex = ''
            light_tex = ''
            diffuse2_offset = None
            diffuse_color = None
            base_color = None
            # 尝试从 shaderParams 中提取（如果存在）
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
            # 变换矩阵：列主序转行主序
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
            # 提取位置：transform 或 pos 字段
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
# Meshopt 解码器（完整）
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
# .mesh 解析（完整所有版本）
# =============================================================================
def read_mesh(data: bytes, filename: str = "") -> Dict[str, Any]:
    if len(data) < 4:
        raise ValueError("文件太小")
    version = read_u32(data, 0)
    name_no_ext = os.path.splitext(os.path.basename(filename))[0]
    if version in (0x17, 0x18):
        return _parse_legacy17(data, name_no_ext, version, filename)
    elif version in (0x19, 0x1A, 0x1B):
        return _parse_legacy1A(data, name_no_ext, version, filename)
    elif version in (0x1C, 0x1D):
        return _parse_legacy1C(data, name_no_ext, version, filename)
    elif version == 0x1E:
        return _parse_legacy1E(data, name_no_ext, version, filename)
    elif version in (0x1F, 0x20):
        return _parse_modern(data, name_no_ext, version, filename)
    else:
        raise ValueError(f"不支持的 mesh 版本: 0x{version:02X}")

def _parse_legacy17(raw: bytes, name: str, version: int, filename: str) -> Dict:
    is_strip = 'StripAnim' in filename
    d = raw
    if is_strip:
        vip, iip, vs = 0x4061, 0x4065, 0x408D
    else:
        p01 = -1
        for i, b in enumerate(d):
            if b == 1:
                p01 = i
                break
        if p01 == -1:
            raise ValueError("v17: 未找到标记")
        vip = p01 + 45
        iip = 0x75
        vs = 0x9D
    vnum = read_u32(d, vip)
    inum = read_u32(d, iip)
    if vnum <= 0 or inum <= 0:
        raise ValueError("v17: 计数无效")
    verts = []
    uvs = []
    for i in range(vnum):
        off = vs + i * 16
        verts.extend(struct.unpack('<fff', d[off:off+12]))
    vbuf_len = vnum * 16
    gap = vbuf_len // 4
    us = vs + vbuf_len + gap
    for i in range(vnum):
        off = us + i * 16
        uvs.append(half_to_float(struct.unpack('<H', d[off:off+2])[0]))
        uvs.append(half_to_float(struct.unpack('<H', d[off+2:off+4])[0]))
    idx_s = us + vbuf_len + vnum * 8 if is_strip else us + vbuf_len
    indices = []
    for i in range(inum // 3):
        off = idx_s + i * 12
        indices.extend(struct.unpack('<III', d[off:off+12]))
    return {'name': name, 'version': version, 'animated': is_strip,
            'vertices': verts, 'uvs': uvs, 'uvs1': None, 'uvs3': None,
            'normals': None, 'indices': indices, 'weighted_vertices': 0,
            'bone_indices': None, 'bone_weights': None, 'skeleton_bones': None, 'bone_count': 0}

def _parse_legacy1A(raw: bytes, name: str, version: int, filename: str) -> Dict:
    d = raw
    vco, ico, vs = 0x66, 0x6A, 0x92
    vnum = read_u32(d, vco)
    inum = read_u32(d, ico)
    if vnum <= 0 or inum <= 0:
        raise ValueError("v1A: 计数无效")
    verts = []
    uvs = []
    for i in range(vnum):
        off = vs + i * 16
        verts.extend(struct.unpack('<fff', d[off:off+12]))
    vbuf_len = vnum * 16
    gap = vbuf_len // 4
    us = vs + vbuf_len + gap
    for i in range(vnum):
        off = us + i * 16
        uvs.append(half_to_float(struct.unpack('<H', d[off:off+2])[0]))
        uvs.append(half_to_float(struct.unpack('<H', d[off+2:off+4])[0]))
    is_sp = ('anim' in filename.lower() or 'anc' in filename.lower()) and 'ancestor' not in filename.lower()
    idx_s = us + vbuf_len + (vnum * 8 if is_sp else 0)
    indices = []
    for i in range(inum // 3):
        off = idx_s + i * 12
        indices.extend(struct.unpack('<III', d[off:off+12]))
    return {'name': name, 'version': version, 'animated': is_sp,
            'vertices': verts, 'uvs': uvs, 'uvs1': None, 'uvs3': None,
            'normals': None, 'indices': indices, 'weighted_vertices': 0,
            'bone_indices': None, 'bone_weights': None, 'skeleton_bones': None, 'bone_count': 0}

def _parse_legacy1C(raw: bytes, name: str, version: int, filename: str) -> Dict:
    d0 = raw
    cs = read_u32(d0, 0x4E)
    us0 = read_u32(d0, 0x52)
    if cs <= 0 or us0 <= 0 or 0x56 + cs > len(d0):
        raise ValueError("v1C: LZ4 边界无效")
    comp = d0[0x56:0x56+cs]
    dr = lz4_decompress(comp, us0)
    d = dr
    vco, ico, vs = 0x34, 0x38, 0x60
    vnum = read_u32(d, vco)
    inum = read_u32(d, ico)
    if vnum <= 0 or inum <= 0:
        raise ValueError("v1C: 计数无效")
    verts = []
    uvs = []
    for i in range(vnum):
        off = vs + i * 16
        verts.extend(struct.unpack('<fff', d[off:off+12]))
    vbuf_len = vnum * 16
    gap = vbuf_len // 4
    us = vs + vbuf_len + gap
    for i in range(vnum):
        off = us + i * 16
        uvs.append(half_to_float(struct.unpack('<H', d[off:off+2])[0]))
        uvs.append(half_to_float(struct.unpack('<H', d[off+2:off+4])[0]))
    is_sp = ('anim' in filename.lower() or 'anc' in filename.lower()) and 'ancestor' not in filename.lower()
    idx_s = us + vbuf_len + (vnum * 8 if is_sp else 0)
    indices = []
    for i in range(inum // 3):
        off = idx_s + i * 12
        indices.extend(struct.unpack('<III', d[off:off+12]))
    return {'name': name, 'version': version, 'animated': is_sp,
            'vertices': verts, 'uvs': uvs, 'uvs1': None, 'uvs3': None,
            'normals': None, 'indices': indices, 'weighted_vertices': 0,
            'bone_indices': None, 'bone_weights': None, 'skeleton_bones': None, 'bone_count': 0}

def _parse_legacy1E(raw: bytes, name: str, version: int, filename: str) -> Dict:
    d0 = raw
    cs = read_u32(d0, 0x4E)
    us0 = read_u32(d0, 0x52)
    if cs <= 0 or us0 <= 0 or 0x56 + cs > len(d0):
        raise ValueError("v1E: LZ4 边界无效")
    comp = d0[0x56:0x56+cs]
    dr = lz4_decompress(comp, us0)
    d = dr
    vnum = read_u32(d, 0x74)
    inum = read_u32(d, 0x78)
    if vnum <= 0 or inum <= 0:
        raise ValueError("v1E: 计数无效")
    vs = 0xB3
    vbuf_len = vnum * 16
    verts = []
    uvs = []
    for i in range(vnum):
        off = vs + i * 16
        verts.extend(struct.unpack('<fff', d[off:off+12]))
    is_sp = 'anim' in filename.lower() or ('anc' in filename.lower() and 'ancestor' not in filename.lower())
    if is_sp:
        gap = vbuf_len // 4
        us = vs + vbuf_len + gap
        uvsz = vbuf_len
        idx_s = us + uvsz + vnum * 8
    else:
        gap = vnum * 4 - 4
        us = vs + vbuf_len + gap
        uvsz = vnum * 16
        idx_s = us + uvsz + 4
    for i in range(vnum):
        off = us + i * 16
        uvs.append(half_to_float(struct.unpack('<H', d[off+4:off+6])[0]))
        uvs.append(half_to_float(struct.unpack('<H', d[off+6:off+8])[0]))
    indices = []
    for i in range(inum // 3):
        off = idx_s + i * 6
        indices.extend(struct.unpack('<HHH', d[off:off+6]))
    return {'name': name, 'version': version, 'animated': is_sp,
            'vertices': verts, 'uvs': uvs, 'uvs1': None, 'uvs3': None,
            'normals': None, 'indices': indices, 'weighted_vertices': 0,
            'bone_indices': None, 'bone_weights': None, 'skeleton_bones': None, 'bone_count': 0}

def _parse_modern(raw: bytes, name: str, version: int, filename: str) -> Dict:
    d = raw
    if len(raw) < 0x58:
        raise ValueError("文件太小")
    animated = d[0x48] != 0
    payload_offset = 0x4E if version >= 0x20 else 0x4A
    is_compressed = struct.unpack('<i', d[payload_offset:payload_offset+4])[0]
    cs = struct.unpack('<i', d[payload_offset+4:payload_offset+8])[0]
    us = struct.unpack('<i', d[payload_offset+8:payload_offset+12])[0]
    if cs <= 0 or us <= 0 or payload_offset + 12 + cs > len(d):
        raise ValueError("压缩数据边界无效")
    src = d[payload_offset+12:payload_offset+12+cs]
    dest = lz4_decompress(src, us) if is_compressed != 0 else src
    skel_start = payload_offset + 12 + cs
    embedded_skeleton_raw = d[skel_start:] if skel_start < len(d) else b''

    p = 4
    def vec3(off):
        return list(struct.unpack('<fff', dest[off:off+12]))
    aabbA = vec3(p); p += 12
    aabbB = vec3(p); p += 12
    aabbA2 = vec3(p); p += 12
    aabbB2 = vec3(p); p += 12
    quant_min = []
    for _ in range(8):
        quant_min.append(struct.unpack('<f', dest[p:p+4])[0]); p += 4
    quant_max = []
    for _ in range(8):
        quant_max.append(struct.unpack('<f', dest[p:p+4])[0]); p += 4
    shared_vertices = struct.unpack('<I', dest[p:p+4])[0]; p += 4
    total_vertices = struct.unpack('<I', dest[p:p+4])[0]; p += 4
    is_idx32 = struct.unpack('<I', dest[p:p+4])[0] != 0; p += 4
    num_points = struct.unpack('<I', dest[p:p+4])[0]; p += 4
    prop11 = struct.unpack('<I', dest[p:p+4])[0]; p += 4
    prop12 = struct.unpack('<I', dest[p:p+4])[0]; p += 4
    prop13 = struct.unpack('<I', dest[p:p+4])[0]; p += 4
    prop14 = struct.unpack('<I', dest[p:p+4])[0]; p += 4
    load_mesh_norms = dest[p] != 0; p += 1
    load_info2 = dest[p] != 0; p += 1
    p += 1
    skip_mesh_pos = struct.unpack('<I', dest[p:p+4])[0]; p += 4
    skip_uvs = struct.unpack('<I', dest[p:p+4])[0]; p += 4
    flag3 = struct.unpack('<I', dest[p:p+4])[0]; p += 4
    p += 0x10

    face_count = total_vertices // 3
    idx_unit = 4 if is_idx32 else 2
    verts = []
    uvs = []
    uvs1 = []
    uvs3 = []

    if skip_mesh_pos == 0:
        for i in range(shared_vertices):
            off = p + i * 16
            verts.extend(struct.unpack('<fff', dest[off:off+12]))
        p += shared_vertices * 16

    norms = None
    if load_mesh_norms:
        norms = [0.0] * (shared_vertices * 3)
        for i in range(shared_vertices):
            off = p + i * 4
            nx = (dest[off] << 24 >> 24) / 127.0
            ny = (dest[off+1] << 24 >> 24) / 127.0
            nz = (dest[off+2] << 24 >> 24) / 127.0
            length = (nx*nx + ny*ny + nz*nz) ** 0.5
            if length == 0:
                length = 1.0
            norms[i*3] = nx / length
            norms[i*3+1] = ny / length
            norms[i*3+2] = nz / length
        p += shared_vertices * 4

    if skip_uvs == 0:
        for i in range(shared_vertices):
            base = p + i * 16
            uvs.append(half_to_float(struct.unpack('<H', dest[base:base+2])[0]))
            uvs.append(half_to_float(struct.unpack('<H', dest[base+2:base+4])[0]))
            uvs1.append(half_to_float(struct.unpack('<H', dest[base+4:base+6])[0]))
            uvs1.append(half_to_float(struct.unpack('<H', dest[base+6:base+8])[0]))
            uvs3.append(half_to_float(struct.unpack('<H', dest[base+12:base+14])[0]))
            uvs3.append(half_to_float(struct.unpack('<H', dest[base+14:base+16])[0]))
        p += shared_vertices * 16

    bone_indices = None
    bone_weights = None
    weighted_vertices = 0
    if animated:
        bone_indices = [0.0] * (shared_vertices * 4)
        bone_weights = [0.0] * (shared_vertices * 4)
        for i in range(shared_vertices):
            off = p + i * 8
            has = False
            for j in range(4):
                bi = dest[off + j]
                wi = dest[off + 4 + j]
                if bi > 0 and wi > 0:
                    bone_indices[i*4 + j] = bi - 1
                    bone_weights[i*4 + j] = wi / 255.0
                    has = True
            if has:
                weighted_vertices += 1
        p += shared_vertices * 8

    indices = []
    for i in range(face_count):
        if is_idx32:
            indices.extend(struct.unpack('<III', dest[p:p+12])); p += 12
        else:
            indices.extend(struct.unpack('<HHH', dest[p:p+6])); p += 6
    if load_info2:
        p += total_vertices * idx_unit
    if num_points > 0:
        p += shared_vertices * idx_unit
    if prop11 > 0:
        p += shared_vertices * idx_unit
    if prop12 > 0:
        p += prop12 * idx_unit
    if prop13 > 0:
        p += prop13 * 4
    if prop14 > 0:
        p += prop14 * (8 if is_idx32 else 4)
    p += face_count * 4

    if skip_mesh_pos > 0:
        ax, ay, az = aabbA2
        sx, sy, sz = aabbB2[0]-ax, aabbB2[1]-ay, aabbB2[2]-az
        for i in range(shared_vertices):
            packed = struct.unpack('<I', dest[p:p+4])[0]; p += 4
            qx = (packed >> 20) & 0x3FF
            qy = (packed >> 10) & 0x3FF
            qz = packed & 0x3FF
            verts.append(ax + (qx / 1023.0) * sx)
            verts.append(ay + (qy / 1023.0) * sy)
            verts.append(az + (qz / 1023.0) * sz)
        p += shared_vertices * 4 + shared_vertices

    if skip_uvs > 0:
        uMinU, uMinV = quant_min[0], quant_min[1]
        uSzU, uSzV = quant_max[0]-uMinU, quant_max[1]-uMinV
        for i in range(shared_vertices):
            off = p + i * 4
            uHi, vHi = dest[off], dest[off+1]
            uLo, vLo = dest[off+2], dest[off+3]
            uN = ((uHi << 8) | uLo) / 65535.0
            vN = ((vHi << 8) | vLo) / 65535.0
            uvs.append(uMinU + uN * uSzU)
            uvs.append(uMinV + vN * uSzV)
        p += shared_vertices * 4

    skeleton_bones = None
    if animated and len(embedded_skeleton_raw) >= 85:
        skeleton_bones = _try_parse_skeleton(embedded_skeleton_raw)

    return {'name': name, 'version': version, 'animated': animated,
            'vertices': verts, 'uvs': uvs, 'uvs1': uvs1 if uvs1 else None,
            'uvs3': uvs3 if uvs3 else None,
            'normals': norms, 'indices': indices,
            'weighted_vertices': weighted_vertices,
            'bone_indices': bone_indices, 'bone_weights': bone_weights,
            'skeleton_bones': skeleton_bones, 'bone_count': len(skeleton_bones) if skeleton_bones else 0}

def _try_parse_skeleton(raw: bytes) -> Optional[List]:
    try:
        return _parse_skeleton(raw)
    except Exception:
        return None

def _parse_skeleton(raw: bytes) -> List:
    p = 0
    p += 4
    p += 64
    num_bones = struct.unpack('<I', raw[p:p+4])[0]; p += 4
    p += 4; p += 4; p += 4; p += 1
    if num_bones <= 0 or num_bones > 4096:
        return []
    bones = []
    for _ in range(num_bones):
        name = read_string(raw, p, 64); p += 64
        mat = list(struct.unpack('<16f', raw[p:p+64])); p += 64
        parent = struct.unpack('<I', raw[p:p+4])[0]; p += 4
        bones.append({'name': name, 'parent': parent-1 if parent > 0 else -1, 'matrix': mat})
    return bones

# =============================================================================
# 地形材质颜色表（移植自 HTML）
# =============================================================================
def get_material_color(mat_idx: int) -> List[float]:
    """获取材质索引对应的 RGB 颜色"""
    MAP_MATERIAL_COLORS = {
        0: [0.5, 0.5, 0.5],
        2: [0.6, 0.7, 0.8],
        3: [0.2, 0.2, 0.2],
        4: [0.9, 0.8, 0.6],
        5: [0.6, 0.45, 0.3],
        6: [0.3, 0.3, 0.3],
        7: [0.65, 0.5, 0.35],
        16: [0.55, 0.5, 0.45],
        17: [0.6, 0.5, 0.35],
        18: [0.65, 0.6, 0.5],
        19: [0.4, 0.35, 0.3],
        20: [0.5, 0.5, 0.55],
        21: [0.9, 0.8, 0.3],
        22: [0.7, 0.85, 0.95],
        23: [0.8, 0.8, 0.85],
        24: [0.75, 0.75, 0.8],
        25: [0.7, 0.7, 0.75],
        26: [0.6, 0.45, 0.35],
        27: [0.4, 0.35, 0.25],
        28: [0.35, 0.35, 0.35],
        29: [0.85, 0.85, 0.8],
        30: [0.6, 0.45, 0.3],
        31: [0.8, 0.7, 0.6],
        32: [0.85, 0.78, 0.55],
        33: [0.7, 0.65, 0.45],
        34: [0.9, 0.85, 0.65],
        35: [0.95, 0.95, 0.98],
        36: [0.75, 0.68, 0.45],
        37: [0.45, 0.4, 0.3],
        48: [0.4, 0.6, 0.3],
        49: [0.3, 0.5, 0.25],
        50: [0.5, 0.7, 0.35],
        51: [0.35, 0.55, 0.3],
        52: [0.7, 0.5, 0.5],
        80: [0.9, 0.9, 1.0],
    }
    return MAP_MATERIAL_COLORS.get(mat_idx, [0.7, 0.7, 0.7])

# =============================================================================
# .meshes 解析（LVL0 地形）
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
        # 法线
        nx = (rv[base+12] << 24 >> 24) / 127.0
        ny = (rv[base+13] << 24 >> 24) / 127.0
        nz = (rv[base+14] << 24 >> 24) / 127.0
        length = (nx*nx + ny*ny + nz*nz)**0.5 or 1.0
        normals.extend([nx/length, ny/length, nz/length])
        
        # ===== 地形顶点颜色（材质混合） =====
        # 4 个材质索引 (b16-b19)
        m0 = rv[base + 16]
        m1 = rv[base + 17]
        m2 = rv[base + 18]
        m3 = rv[base + 19]
        # 4 个混合权重 (b20-b23) 归一化到 0-1
        w0 = rv[base + 20] / 255.0
        w1 = rv[base + 21] / 255.0
        w2 = rv[base + 22] / 255.0
        w3 = rv[base + 23] / 255.0
        tw = w0 + w1 + w2 + w3
        
        if tw < 0.001:
            # 无权重，直接用第一个材质
            c = get_material_color(m0)
            colors.extend(c)
        else:
            c0 = get_material_color(m0)
            c1 = get_material_color(m1)
            c2 = get_material_color(m2)
            c3 = get_material_color(m3)
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
def convert_map(map_folder: str, mesh_root: str, output_root: str, fmt: str = 'glb', export_markers: bool = True):
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

    # 解析地形
    try:
        terrain_data = parse_meshes(meshes_file.read_bytes())
        print(f"地形: 顶点 {terrain_data['vertex_count']}, 面 {len(terrain_data['indices'])//3}, 云面 {len(terrain_data['cloud_indices'])//3}")
    except Exception as e:
        print(f"地形解析失败: {e}")
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
    pos = np.array(terrain_data['positions']).reshape(-1, 3)
    norm = np.array(terrain_data['normals']).reshape(-1, 3)
    col = np.array(terrain_data['colors']).reshape(-1, 3)
    idx = np.array(terrain_data['indices'], dtype=np.uint32)
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
        # 应用变换矩阵
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

    # ===== 4. 标记点 → 小圆球（修正索引） =====
    if export_markers and markers:
        import math

        # 球体参数
        MARKER_RADIUS = 0.5
        SEGMENTS = 8  # 经线数，越大越圆
        rings = SEGMENTS // 2  # 纬线环数

        # 预生成球体顶点和索引（只生成一次，所有标记点共用几何）
        sphere_verts = []
        sphere_faces = []

        # 北极点
        sphere_verts.append((0, MARKER_RADIUS, 0))
        # 南极点
        sphere_verts.append((0, -MARKER_RADIUS, 0))

        # 中间纬线环
        for i in range(1, rings):
            phi = math.pi * i / rings
            y = MARKER_RADIUS * math.cos(phi)
            r = MARKER_RADIUS * math.sin(phi)
            for j in range(SEGMENTS):
                theta = 2 * math.pi * j / SEGMENTS
                x = r * math.cos(theta)
                z = r * math.sin(theta)
                sphere_verts.append((x, y, z))

        # 北极三角扇
        for j in range(SEGMENTS):
            sphere_faces.append((0, 2 + j, 2 + (j + 1) % SEGMENTS))

        # 南极三角扇（修正 base_idx）
        base_idx = 2 + (rings - 2) * SEGMENTS
        for j in range(SEGMENTS):
            sphere_faces.append((1, base_idx + (j + 1) % SEGMENTS, base_idx + j))

        # 中间四边形网格（拆成两个三角形）
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

        print(f"标记点球体半径: {MARKER_RADIUS}，分段数: {SEGMENTS}，顶点数: {len(sphere_verts)}，面数: {len(sphere_faces)//3}")

        for group in markers:
            color_hex = group['color'].lstrip('#')
            r = int(color_hex[0:2], 16) / 255.0
            g = int(color_hex[2:4], 16) / 255.0
            b = int(color_hex[4:6], 16) / 255.0
            color_rgb = np.array([r, g, b])

            for pt in group['points']:
                pos = np.array(pt['pos'])
                # 平移球体顶点到目标位置
                verts = sphere_verts + pos
                # 顶点颜色（每个顶点同色）
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
    print("=== Sky 地图转模型工具（全融合版） ===")
    print("支持所有 .mesh 版本（0x17~0x20）及地图 .meshes")
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

    export_markers = input("是否导出标记点？ (y/n, 默认 y): ").strip().lower() != 'n'

    for map_name, folder in map_folders:
        try:
            convert_map(folder, mesh_root, output_root, fmt, export_markers)
        except Exception as e:
            print(f"转换 {folder} 失败: {e}")
            import traceback
            traceback.print_exc()
    print("\nok")

if __name__ == '__main__':
    main()