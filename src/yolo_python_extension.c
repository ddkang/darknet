#include <Python.h> // must be included before all other headers to avoid issues
#include <numpy/arrayobject.h> 
#include "structmember.h"

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

#define is_array(a)            ((a) && PyArray_Check((PyArrayObject *)a))
#define array_type(a)          (int)(PyArray_TYPE(a))
#define array_numdims(a)       (((PyArrayObject *)a)->nd)
#define array_dimensions(a)    (((PyArrayObject *)a)->dimensions)
#define array_size(a,i)        (((PyArrayObject *)a)->dimensions[i])
#define array_data(a)          (((PyArrayObject *)a)->data)
#define array_is_contiguous(a) (PyArray_ISCONTIGUOUS(a))
#define array_is_native(a)     (PyArray_ISNOTSWAPPED(a))
#define array_is_fortran(a)    (PyArray_ISFORTRAN(a))

////////////////////////////////////////////////////////////////////////////////
// yolo python wrapper module
////////////////////////////////////////////////////////////////////////////////
typedef struct {
  // Python garbage
  PyObject_HEAD

  // Detection parameters
  float thresh;
  float nms;

  image **alphabet;
  char **names;
  int classes;

  network net;

  box *boxes;
  float **probs;
} YOLO;

static void YOLO_dealloc(YOLO* self) {
  int j;
  layer l = self->net.layers[self->net.n-1];
  for(j = 0; j < l.w * l.h * l.n; ++j) {
    free(self->probs[j]);
  }
  for (j = 0; j < self->classes; ++j) {
    free(self->names[j]);
  }
  free(self->probs);
  free(self->boxes);

  self->ob_type->tp_free((PyObject*)self);
}

static PyObject * YOLO_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
  YOLO *self = (YOLO *)type->tp_alloc(type, 0);
  return (PyObject *)self;
}

static int YOLO_init(YOLO *self, PyObject *args, PyObject *kwds) {
  int j;
  char* cfgfile;
  char* weightfile;
  char* labelsdir;
  char* datacfg;

  // parse input arguments
  PyObject *config_filename, *weights_filename, *data_config, *tmp;

  static char *kwlist[] = {"config_filename", "weights_filename", "data_config", NULL};
  if (!PyArg_ParseTupleAndKeywords(
       args, kwds, "SSS", kwlist,
       &config_filename, &weights_filename, &data_config)) {
    return NULL; 
  }

  // get char* representations of input arguments
  cfgfile = PyString_AsString(config_filename);
  weightfile = PyString_AsString(weights_filename);
  datacfg = PyString_AsString(data_config);

  list *options = read_data_cfg(datacfg);
  // FIXME
  char *name_list = option_find_str(options, "names", "~/tmp/vuse/vuse/darknet/data/names.list");
  self->classes = option_find_int(options, "classes", 20);
  self->names = get_labels(name_list);

  // setup yolo network and load weights
  self->nms = 0.4;
  self->thresh = 0.2;

  fprintf(stderr, "---loading alphabet\n");
  // self->alphabet = load_alphabet_dirname(labelsdir);
  fprintf(stderr, "---done loading alphabet!\n");


  fprintf(stderr, "---parsing network\n");
  self->net = parse_network_cfg(cfgfile);
  fprintf(stderr, "---done parsing network!\n");


  fprintf(stderr, "---loading weights\n");
  load_weights(&self->net, weightfile);
  fprintf(stderr, "---done loading weights!\n");


  fprintf(stderr, "---allocating memory\n");
  set_batch_network(&self->net, 1);
  srand(2222222);
  layer l = self->net.layers[self->net.n - 1];
  self->boxes = calloc(l.w * l.h * l.n, sizeof(box));
  self->probs = calloc(l.w * l.h * l.n, sizeof(float *));
  for (j = 0; j < l.w * l.h * l.n; ++j) {
    self->probs[j] = calloc(l.classes, sizeof(float *));
  }  
  fprintf(stderr, "---done allocating memory!\n");
  
  return 0;
}

static PyGetSetDef YOLO_getseters[] = {
  {NULL}  /* Sentinel */
};

// FIXME: This shares a lot of code with YOLO standalone.
// Combine them?
static PyObject * YOLO_label_frame(YOLO* self, PyObject *args) {
  static PyObject *format = NULL;
  PyArrayObject *frame;
  char* input;
  image im, sized;
  float *X;
  int i, j, k;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &frame)) {
    return NULL;
  }
  //fprintf(stderr, "---type of array %i\n", is_array(frame));

  //fprintf(stderr, "dims: %i\n", array_numdims(frame));
  //fprintf(stderr, "dim0: %i\n", array_size(frame, 0));
  //fprintf(stderr, "dim1: %i\n", array_size(frame, 1));
  //fprintf(stderr, "dim2: %i\n", array_size(frame, 2));
  //fprintf(stderr, "is_contiguous: %i\n", array_is_contiguous(frame));
  //fprintf(stderr, "type (uint8): %i\n", array_type(frame) == NPY_UINT8);

  // memory layout [r1, g1, b1, r2, g2, b2, ... ] goes row by row
  const uint8_t *data = (uint8_t*) array_data(frame);

  const int w = array_size(frame, 1);
  const int h = array_size(frame, 0);
  const int c = array_size(frame, 2);
  im = make_image(w, h, c);

  for (k = 0; k < c; ++k) {
    for (j = 0; j < h; ++j) {
      for (i = 0; i < w; ++i) {
        int dst_index = i + w*j + w*h*k;
        int src_index = k + c*i + c*w*j;
        im.data[dst_index] = (float)data[src_index]/255.0;
      }
    }
  }

  sized = resize_image(im, self->net.w, self->net.h);
  X = sized.data;

  network_predict(self->net, X);
  layer l = self->net.layers[self->net.n-1];
  if (l.type == DETECTION) {
    get_detection_boxes(l, 1, 1, self->thresh, self->probs, self->boxes, 0);
  } else if (l.type == REGION) {
    get_region_boxes(l, 1, 1, self->thresh, self->probs, self->boxes, 0, 0, self->thresh);
  } else {
    fprintf(stderr, "Last layer must be detections.\n");
    exit(1);
  }
  if (self->nms > 0) do_nms(self->boxes, self->probs, l.w*l.h*l.n, l.classes, self->nms);

  PyObject* detections_list = PyList_New(0);
  for (j = 0; j < l.w * l.h * l.n; j++) {
    for (k = 0; k < l.classes; k++) {
      if (self->probs[j][k] > self->thresh) {
        const float xmin = self->boxes[j].x - self->boxes[j].w/2.;
        const float xmax = self->boxes[j].x + self->boxes[j].w/2.;
        const float ymin = self->boxes[j].y - self->boxes[j].h/2.;
        const float ymax = self->boxes[j].y + self->boxes[j].h/2.;

        // add detections to dictionary
        PyObject* detection_dict = PyDict_New();

        PyObject* object_name_val = PyString_FromString(self->names[k]);
        if( PyDict_SetItemString(detection_dict, "object_name", object_name_val) ) {
          return NULL;
        }

        PyObject* object_confidence_val = PyFloat_FromDouble((double)self->probs[j][k]);
        if( PyDict_SetItemString(detection_dict, "confidence", object_confidence_val) ) {
          return NULL;
        }

        PyObject* xmin_val = PyFloat_FromDouble((double)xmin);
        if( PyDict_SetItemString(detection_dict, "xmin", xmin_val) ) {
          return NULL;
        }        

        PyObject* xmax_val = PyFloat_FromDouble((double)xmax);
        if( PyDict_SetItemString(detection_dict, "xmax", xmax_val) ) {
          return NULL;
        }        


        PyObject* ymin_val = PyFloat_FromDouble((double)ymin);
        if( PyDict_SetItemString(detection_dict, "ymin", ymin_val) ) {
          return NULL;
        }        

        PyObject* ymax_val = PyFloat_FromDouble((double)ymax);
        if( PyDict_SetItemString(detection_dict, "ymax", ymax_val) ) {
          return NULL;
        }        

        // append dictionary to the detections list
        if( PyList_Append(detections_list, detection_dict) ){
          return NULL;
        }

      }
    }
  }

  free_image(im);
  free_image(sized);

  // return the list of detected objects
  return detections_list;
}

static PyMethodDef YOLO_methods[] = {
  {"label_frame", (PyCFunction)YOLO_label_frame, METH_VARARGS,
     "Returns labels for objects in the frame by pushing the frame through the YOLO CNN."
  },
  {NULL}  /* Sentinel */
};

static PyTypeObject YOLOType = {
  PyObject_HEAD_INIT(NULL)
  0,                         /*ob_size*/
  "vuse_yolo.YOLO",          /*tp_name*/
  sizeof(YOLO),              /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  (destructor)YOLO_dealloc,  /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "YOLO objects",            /* tp_doc */
  0,                         /* tp_traverse */
  0,                         /* tp_clear */
  0,                         /* tp_richcompare */
  0,                         /* tp_weaklistoffset */
  0,                         /* tp_iter */
  0,                         /* tp_iternext */
  YOLO_methods,              /* tp_methods */
  0,                         /* tp_members */
  YOLO_getseters,            /* tp_getset */
  0,                         /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  (initproc)YOLO_init,       /* tp_init */
  0,                         /* tp_alloc */
  YOLO_new,                  /* tp_new */
};

static PyMethodDef module_methods[] = {
  {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initvuse_yolo(void) {
  PyObject* m;

  if (PyType_Ready(&YOLOType) < 0) {
    return;
  }

  m = Py_InitModule3("vuse_yolo",
                     module_methods,
                     "Python wrapper around the YOLO CNN-based object detector."
                     );

  if (m == NULL) {
    return;
  }

  Py_INCREF(&YOLOType);
  PyModule_AddObject(m, "YOLO", (PyObject *)&YOLOType);
  
  import_array(); // DO NOT REMOVE! This is vital to numpy working correctly.
}
