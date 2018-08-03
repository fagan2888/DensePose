# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

import tornado
import tornado.ioloop
import tornado.web
import tempfile
import shutil
import base64


logger = logging.getLogger(__name__)


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--port',
        help='service port',
        default=8888,
        type=int
    )
    parser.add_argument(
        '--public-dir',
        help='public directory',
        default='server/public/',
        type=str
    )
    return parser.parse_args()


def detect(model, in_file, dummy_coco_dataset):
    # logger.info('Processing {} -> {}'.format(in_file, out_file))
    im = cv2.imread(in_file)
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    for k, v in timers.items():
        logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

    output_dir = tempfile.mkdtemp()
    try:
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            in_file,
            output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2,
            ext="png",
        )
        suffixes = {'inds': '_INDS.png', 'iuv': '_IUV.png'}
        results = {}
        for suffix in suffixes:
            output_name = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(in_file))[0] + suffixes[suffix]
            )
            with open(output_name, 'rb') as f:
                results[suffix] = base64.b64encode(f.read())
    finally:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    return dict(
        cls_boxes=cls_boxes,
        cls_segms=cls_segms,
        cls_keyps=cls_keyps,
        cls_bodys=cls_bodys,
        visualization=results,
        timers={k: v.average_time for k, v in timers.items()},
    )


class Userform(tornado.web.RequestHandler):
    def initialize(self, filename):
        self.filename = filename

    def get(self):
        self.render(self.filename)


class Upload(tornado.web.RequestHandler):
    def initialize(self, model):
        self.model = model
        self.dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    def post(self):
        fileinfo = self.request.files['file1'][0]
        with tempfile.NamedTemporaryFile() as f:
            f.write(fileinfo['body'])
            f.flush()
            result = detect(self.model, f.name, self.dummy_coco_dataset)
        # self.write(result['timers'])
        self.write(result['visualization'])


def create_app(args, model):
    return tornado.web.Application([
            (r"/", Userform, dict(
                filename=os.path.join(args.public_dir, "index.html"),
            )),
            (r"/upload", Upload, dict(
                model=model,
            )),
            (r"/static/(.*)", tornado.web.StaticFileHandler, dict(
                path=os.path.join(args.public_dir, "static"),
            )),
        ],
        debug=True
    )


def main(args):
    assert os.path.exists(args.public_dir)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)

    app = create_app(args, model)
    app.listen(args.port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
