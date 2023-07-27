#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

import ctypes
import numpy as np

import pyds

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

MAX_OUTPUT_BBOX_COUNT = 100
INPUT_H = 384
INPUT_W = 640

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    # NvDsUserMeta
    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.glist_get_nvds_frame_meta()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            #frame_meta = pyds.glist_get_nvds_frame_meta(l_frame.data)
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if (user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):
                continue

            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

            # Boxes in the tensor meta should be in network resolution which is
            # found in tensor_meta.network_info. Use this info to scale boxes to
            # the input frame resolution.
            # print("YF: length of layers:", tensor_meta.num_output_layers)  # 一共有四个输出prob/boxes/classes/points，顺序对应yolov5.cpp中的输出
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
            # print("YF: layer name:", layer.layerName)
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            prob = np.ctypeslib.as_array(ptr, shape=(MAX_OUTPUT_BBOX_COUNT,))
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 1)
            # print("YF: layer name:", layer.layerName)
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            boxes = np.ctypeslib.as_array(ptr, shape=(MAX_OUTPUT_BBOX_COUNT, 4))
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 2)
            # print("YF: layer name:", layer.layerName)
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            classes = np.ctypeslib.as_array(ptr, shape=(MAX_OUTPUT_BBOX_COUNT,))
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 3)
            # print("YF: layer name:", layer.layerName)
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            points = np.ctypeslib.as_array(ptr, shape=(MAX_OUTPUT_BBOX_COUNT, 17, 2))
            
            for s, b, c, p in zip(prob, boxes, classes, points):
                if s < 0.1: continue 
                
                """ Inserts an object into the metadata """
                # this is a good place to insert objects into the metadata.
                # Here's an example of inserting a single object.
                obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                # Set bbox properties. These are in input resolution.
                rect_params = obj_meta.rect_params
                rect_params.left = int(b[0])
                rect_params.top = int(b[1])
                rect_params.width = int(b[2]-b[0])
                rect_params.height = int(b[3]-b[1])

                # Semi-transparent yellow backgroud
                rect_params.has_bg_color = 0
                rect_params.bg_color.set(1, 1, 0, 0.4)

                # Red border of width 3
                rect_params.border_width = 3
                rect_params.border_color.set(1, 0, 0, 1)

                # Set object info including class, detection confidence, etc.
                obj_meta.confidence = s
                obj_meta.class_id = c

                # There is no tracking ID upon detection. The tracker will
                # assign an ID.
                obj_meta.object_id = 0
                # Set the object classification label.
                obj_meta.obj_label = "{}".format(c)

                # Set display text for the object.
                txt_params = obj_meta.text_params
                if txt_params.display_text:
                    pyds.free_buffer(txt_params.display_text)

                txt_params.x_offset = int(rect_params.left)
                txt_params.y_offset = max(0, int(rect_params.top) - 10)
                txt_params.display_text = (
                    "{}".format(c) + " " + "{:04.3f}".format(s)
                )
                # Font , font-color and font-size
                txt_params.font_params.font_name = "Serif"
                txt_params.font_params.font_size = 10
                # set(red, green, blue, alpha); set to White
                txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

                # Text background color
                txt_params.set_bg_clr = 1
                # set(red, green, blue, alpha); set to Black
                txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

                # Inser the object into current frame meta
                # This object has no parent
                pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)
                        
                display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                display_meta.num_circles = 16  # cicle一次只能添加16个点, 假如想多个，只能修改MAX_ELEMENTS_IN_DISPLAY_META的值，然后重新编译DS
                circle_params = display_meta.circle_params
                for i in range(16):
                    circle_params[i].xc = int(p[i, 0])
                    circle_params[i].yc = int(p[i, 1])
                    circle_params[i].radius = 2
                    circle_params[i].circle_color.set(1.0, 1.0, 1.0, 1.0)
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
                 
            # good luck !!!!
            print("good luck")
            
            try:
                l_user = l_user.next
            except StopIteration:
                break

        try:
            # indicate inference is performed on the frame
            frame_meta.bInferDone = True
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK	


def make_elm_or_print_err(factoryname, name, printedname, detail=""):
    """ Creates an element with Gst Element Factory make.
        Return the element  if successfully created, otherwise print
        to stderr and return None.
    """
    print("Creating", printedname)
    elm = Gst.ElementFactory.make(factoryname, name)
    if not elm:
        sys.stderr.write("Unable to create " + printedname + " \n")
        if detail:
            sys.stderr.write(detail)
    return elm

def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    print("Creating H264Parser \n")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write(" Unable to create h264 parser \n")

    # Use nvdec_h264 for hardware accelerated decode on GPU
    print("Creating Decoder \n")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # # Finally render the osd output
    # if is_aarch64():
    #     transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")

    # print("Creating EGLSink \n")
    # sink = Gst.ElementFactory.make("fakesink", "nvvideo-renderer")
    # if not sink:
    #     sys.stderr.write(" Unable to create egl sink \n")
     # Finally encode and save the osd output
    queue = make_elm_or_print_err("queue", "queue", "Queue")

    nvvidconv2 = make_elm_or_print_err("nvvideoconvert", "convertor2", "Converter 2 (nvvidconv2)")

    capsfilter = make_elm_or_print_err("capsfilter", "capsfilter", "capsfilter")

    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)

    # On Jetson, there is a problem with the encoder failing to initialize
    # due to limitation on TLS usage. To work around this, preload libgomp.
    # Add a reminder here in case the user forgets.
    preload_reminder = "If the following error is encountered:\n" + \
                       "/usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block\n" + \
                       "Preload the offending library:\n" + \
                       "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1\n"
    encoder = make_elm_or_print_err("avenc_mpeg4", "encoder", "Encoder", preload_reminder)

    encoder.set_property("bitrate", 2000000)

    codeparser = make_elm_or_print_err("mpeg4videoparse", "mpeg4-parser", 'Code Parser')

    container = make_elm_or_print_err("qtmux", "qtmux", "Container")

    sink = make_elm_or_print_err("filesink", "filesink", "Sink")
    sink.set_property("location", "out.mp4")
    sink.set_property("sync", 0)
    sink.set_property("async", 0)


    print("Playing file %s " %args[1])
    source.set_property('location', args[1])
    streammux.set_property('width', INPUT_W)
    streammux.set_property('height', INPUT_H)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    # pgie.set_property('config-file-path', "dstest1_pgie_config.txt")
    pgie.set_property('config-file-path', "yolov5_pose.txt")

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    # pipeline.add(sink)
    # if is_aarch64():
    #     pipeline.add(transform)
    pipeline.add(queue)
    pipeline.add(nvvidconv2)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(codeparser)
    pipeline.add(container)
    pipeline.add(sink)

    # we link the elements together
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    print("Linking elements in the Pipeline \n")
    source.link(h264parser)
    h264parser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    # if is_aarch64():
    #     nvosd.link(transform)
    #     transform.link(sink)
    # else:
        # nvosd.link(sink)
    nvosd.link(queue)
    queue.link(nvvidconv2)
    nvvidconv2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(codeparser)
    codeparser.link(container)
    container.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

