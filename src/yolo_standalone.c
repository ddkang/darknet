#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "region_layer.h"

#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/videoio/videoio_c.h"
image get_image_from_stream(CvCapture *cap);


typedef struct {
  float thresh;
  float nms;

  image **alphabet;
  char **names;
  int classes;

  network net;

  box *boxes;
  float **probs;

  FILE *fout;
} YOLO;

static void YOLO_dealloc(YOLO* self) {
  int j;
  layer l = self->net.layers[self->net.n-1];
  for (j = 0; j < l.w * l.h * l.n; ++j) {
    free(self->probs[j]);
  }
  for (j = 0; j < self->classes; j++) {
    free(self->names[j]);
  }
  free(self->probs);
  free(self->boxes);
  free(self->names);
}

static int YOLO_init(YOLO *self, char *datacfg, char *cfgfile, char *weightfile, char *labelsdir,
    char *csv_file) {
  int j;
  layer l;

  list *options = read_data_cfg(datacfg);
  char *name_list = option_find_str(options, "names", "data/names.list");
  self->classes = option_find_int(options, "classes", 20);
  self->names = get_labels(name_list);

  self->fout = fopen(csv_file, "wb");
  fprintf(self->fout, "frame,object_name,confidence,xmin,ymin,xmax,ymax\n");

  // setup yolo network and load weights
  self->nms = 0.4;
  self->thresh = 0.2;

  fprintf(stderr, "---loading alphabet\n");
  self->alphabet = load_alphabet();
  fprintf(stderr, "---done loading alphabet!\n");

  fprintf(stderr, "---parsing network\n");
  self->net = parse_network_cfg(cfgfile);
  fprintf(stderr, "---done parsing network!\n");

  fprintf(stderr, "---loading weights\n");
  load_weights(&self->net, weightfile);
  fprintf(stderr, "---done loading weights!\n");

  fprintf(stderr, "---allocating memory\n");
  set_batch_network(&self->net, 1);
  l = self->net.layers[self->net.n - 1];
  srand(2222222);
  self->boxes = (box *) calloc(l.w * l.h * l.n, sizeof(box));
  self->probs = (float **) calloc(l.w * l.h * l.n, sizeof(float *));
  for (j = 0; j < l.w * l.h * l.n; j++)
    self->probs[j] = (float *) calloc(l.classes, sizeof(float));
  fprintf(stderr, "---done allocating memory!\n");

  return 0;
}

static int YOLO_label_frame(YOLO *self, image im, int frame_num) {
  image sized = resize_image(im, self->net.w, self->net.h);
  float *X = sized.data;
  layer l = self->net.layers[self->net.n-1];
  int j, k, printed = false;

  network_predict(self->net, X);
  if (l.type == DETECTION) {
    get_detection_boxes(l, 1, 1, self->thresh, self->probs, self->boxes, 0);
  } else if (l.type == REGION) {
		int *map = 0;
    get_region_boxes(l, im.w, im.h, self->net.w, self->net.h, self->thresh, self->probs, self->boxes, 0, 0, map, self->thresh, 0);
  } else {
    fprintf(stderr, "Last layer must be detections.\n");
    exit(1);
  }
  if (self->nms > 0) do_nms(self->boxes, self->probs, l.w*l.h*l.n, l.classes, self->nms);

  for (j = 0; j < l.w * l.h * l.n; j++) {
    for (k = 0; k < l.classes; k++) {
      if (self->probs[j][k] > self->thresh) {
        float xmin = self->boxes[j].x - self->boxes[j].w/2.;
        float xmax = self->boxes[j].x + self->boxes[j].w/2.;
        float ymin = self->boxes[j].y - self->boxes[j].h/2.;
        float ymax = self->boxes[j].y + self->boxes[j].h/2.;
        float confidence = self->probs[j][k];
        char *object_name = self->names[k];

        // frame,object_name,confidence,xmin,ymin,xmax,ymax
        fprintf(self->fout, "%d,%s,%f,%f,%f,%f,%f\n",
            frame_num, object_name, confidence,
            xmin, ymin, xmax, ymax);
      }
    }
  }

  free_image(sized);

  return 0;
}

int main(int argc, char **argv) {
  // FIXME
  char data_cfg[] = "cfg/coco.data";
  char cfg_file[] = "cfg/yolo.cfg";
  char weight_file[] = "weights/yolo.weights";
  char labels_dir[] = "data/labels";
  char *video_file, *csv_file;

  // Parse command line arguments.
  // Not a fan of positional arguments, but oh well
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <video file> <output csv> <start_frame>\n", argv[0]);
    exit(0);
  }
  video_file = argv[1];
  csv_file = argv[2];
  const int start_frame = atoi(argv[3]);

  // Open video stream
  CvCapture *cap = cvCaptureFromFile(video_file);
  if (!cap) {
    fprintf(stderr, "Video file failed to init.\n");
    exit(1);
  }

  // Init yolo
  YOLO *yolo = (YOLO *) calloc(1, sizeof(YOLO));
  YOLO_init(yolo, data_cfg, cfg_file, weight_file, labels_dir, csv_file);

  // Label images
  cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, start_frame);
  for (int i = start_frame; i < start_frame + 300000; i++) {
    if (i % 500 == 0)
      fprintf(stderr, "frame %d\n", i);
    image im = get_image_from_stream(cap);
    if (!im.data) {
      fprintf(stderr, "Stream closed.\n");
      break;
    }
    YOLO_label_frame(yolo, im, i);
    free_image(im);
  }

  YOLO_dealloc(yolo);
  // cvReleaseCapture(cap);
  return 0;
}
