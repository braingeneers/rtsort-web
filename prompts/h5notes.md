i(venv) rcurrie at macbook in ~/rtsort-web on main\*
$ npm run dev

> rtsort-web@0.0.0 dev
> vite

VITE v6.3.5 ready in 117 ms

➜ Local: http://localhost:5173/rtsort-web/
➜ Network: use --host to expose
➜ press h + enter to show help
[vite-plugin-static-copy] Collected 13 items.
3:03:17 PM [vite] (client) page reload src/worker.ts
3:05:45 PM [vite] (client) page reload src/worker.ts (x2)
3:05:54 PM [vite] (client) page reload src/worker.ts (x3)
3:06:19 PM [vite] (client) page reload src/worker.ts (x4)
3:07:27 PM [vite] (client) page reload src/worker.ts (x5)
3:08:18 PM [vite] (client) page reload src/worker.ts (x6)
3:13:34 PM [vite] (client) page reload src/worker.ts (x7)
3:13:44 PM [vite] (client) page reload src/worker.ts (x8)
3:15:06 PM [vite] (client) page reload src/worker.ts (x9)
3:16:03 PM [vite] (client) page reload src/worker.ts (x10)
3:16:56 PM [vite] (client) page reload src/worker.ts (x11)
3:17:47 PM [vite] (client) page reload src/worker.ts (x12)
3:18:50 PM [vite] (client) page reload src/worker.ts (x13)
3:18:50 PM [vite] (client) Pre-transform error: Transform failed with 1 error:
/Users/rcurrie/rtsort-web/src/worker.ts?worker:85:2: ERROR: Unexpected "for"
Plugin: vite:esbuild
File: /Users/rcurrie/rtsort-web/src/worker.ts?worker:85:2

Unexpected "for"
83 |  
 84 | // Process windows
85 | for (let startFrame = 0; startFrame <= totalSamples - sampleSize; startFrame += numOutputLocs) {
| ^
86 | // Extract and process window data for each channel
87 | const processedData = new Float16Array(numChannels \* sampleSize)

3:18:51 PM [vite] Internal server error: Transform failed with 1 error:
/Users/rcurrie/rtsort-web/src/worker.ts?worker:85:2: ERROR: Unexpected "for"
Plugin: vite:esbuild
File: /Users/rcurrie/rtsort-web/src/worker.ts?worker:85:2

Unexpected "for"
83 |  
 84 | // Process windows
85 | for (let startFrame = 0; startFrame <= totalSamples - sampleSize; startFrame += numOutputLocs) {
| ^
86 | // Extract and process window data for each channel
87 | const processedData = new Float16Array(numChannels \* sampleSize)

      at failureErrorWithLog (/Users/rcurrie/rtsort-web/node_modules/esbuild/lib/main.js:1463:15)
      at /Users/rcurrie/rtsort-web/node_modules/esbuild/lib/main.js:734:50
      at responseCallbacks.<computed> (/Users/rcurrie/rtsort-web/node_modules/esbuild/lib/main.js:601:9)
      at handleIncomingPacket (/Users/rcurrie/rtsort-web/node_modules/esbuild/lib/main.js:656:12)
      at Socket.readFromStdout (/Users/rcurrie/rtsort-web/node_modules/esbuild/lib/main.js:579:7)
      at Socket.emit (node:events:518:28)
      at addChunk (node:internal/streams/readable:561:12)
      at readableAddChunkPushByteMode (node:internal/streams/readable:512:3)
      at Readable.push (node:internal/streams/readable:392:5)
      at Pipe.onStreamRead (node:internal/stream_base_commons:189:23)

3:18:56 PM [vite] (client) page reload src/worker.ts
3:22:32 PM [vite] (client) page reload public/test.html
3:22:42 PM [vite] (client) page reload test-inference-scaling.html
3:22:44 PM [vite] (client) page reload test-worker.html
3:26:13 PM [vite] (client) page reload src/worker.ts
3:26:40 PM [vite] (client) page reload src/worker.ts (x2)
3:26:55 PM [vite] (client) page reload src/worker.ts (x3)
3:27:22 PM [vite] (client) page reload src/worker.ts (x4)
3:27:52 PM [vite] (client) page reload src/worker.ts (x5)
3:28:36 PM [vite] (client) page reload src/worker.ts (x6)
5:08:29 PM [vite] [vite-plugin-static-copy] detected new file public/models/model_outputs.npy
5:08:29 PM [vite] [vite-plugin-static-copy] detected new file public/models/scaled_traces.npy
5:19:58 PM [vite] (client) page reload src/worker.ts
5:20:14 PM [vite] (client) page reload src/worker.ts (x2)
5:21:40 PM [vite] (client) page reload src/worker.ts (x3)
5:28:15 PM [vite] (client) hmr update /src/App.vue
5:30:43 PM [vite] (client) hmr update /src/App.vue (x2)
5:37:04 PM [vite] (client) hmr update /src/App.vue (x3)
5:39:16 PM [vite] (client) hmr update /src/App.vue (x4)
6:09:04 PM [vite] (client) page reload src/worker.ts
6:09:17 PM [vite] (client) page reload src/worker.ts (x2)
6:09:50 PM [vite] (client) page reload src/worker.ts (x3)
6:10:16 PM [vite] (client) page reload src/worker.ts (x4)
6:10:34 PM [vite] (client) hmr update /src/App.vue
6:10:41 PM [vite] (client) hmr update /src/App.vue (x2)
6:10:57 PM [vite] (client) hmr update /src/App.vue (x3)
6:11:06 PM [vite] (client) hmr update /src/App.vue (x4)
6:11:16 PM [vite] (client) hmr update /src/App.vue (x5)
6:12:33 PM [vite] (client) page reload src/worker.ts
6:12:40 PM [vite] (client) hmr update /src/App.vue
6:13:10 PM [vite] (client) hmr update /src/App.vue (x2)
^C
(venv) rcurrie at macbook in ~/rtsort-web on main\*
$ npm run dev

> rtsort-web@0.0.0 dev
> vite

VITE v6.3.5 ready in 175 ms

➜ Local: http://localhost:5173/rtsort-web/
➜ Network: use --host to expose
➜ press h + enter to show help
[vite-plugin-static-copy] Collected 14 items.
^C
(venv) rcurrie at macbook in ~/rtsort-web on main\*
$ ipython
Python 3.11.12 (main, Apr 8 2025, 14:15:29) [Clang 16.0.0 (clang-1600.0.26.6)]
Type 'copyright', 'credits' or 'license' for more information
IPython 9.2.0 -- An enhanced Interactive Python. Type '?' for help.
Tip: You can use Ctrl-O to force a new line in terminal IPython

In [1]: from spikeinterface.extractors import MaxwellRecordingExtractor

In [2]: r = MaxwellRecordingExtractor("data/MEA_rec_patch_ground_truth_cell7.raw.h5")

In [3]: r
Out[3]:
MaxwellRecordingExtractor: 942 channels - 20.0kHz - 1 segments - 3,723,600 samples
186.18s (3.10 minutes) - uint16 dtype - 6.53 GiB
file_path: /Users/rcurrie/rtsort-web/data/MEA_rec_patch_ground_truth_cell7.raw.h5

In [4]: import numpy as np

In [5]: a = np.load("data/scaled_traces.npy")

In [6]: a.dtype
Out[6]: dtype('float16')

In [7]: a
Out[7]:
array([[3228., 3248., 3204., ..., 3222., 3198., 3242.],
       [3198., 3216., 3178., ..., 3192., 3166., 3192.],
       [3140., 3154., 3110., ..., 3198., 3166., 3210.],
       ...,
       [3198., 3216., 3172., ..., 3192., 3172., 3192.],
       [3160., 3172., 3134., ..., 3184., 3166., 3204.],
       [3228., 3254., 3178., ..., 3248., 3216., 3280.]],
shape=(942, 100000), dtype=float16)

In [8]: r.has_scaled_traces()
<ipython-input-8-000970189a9b>:1: DeprecationWarning: `has_scaled_traces` is deprecated and will be removed in 0.103.0. Use has_scaleable_traces() instead
r.has_scaled_traces()
Out[8]: True

In [9]: r.has_scaleable_traces()
Out[9]: True

In [10]: import h5py

In [11]: h = h5py.File("data/MEA_rec_patch_ground_truth_cell7.raw.h5", "r")

In [12]: h
Out[12]: <HDF5 file "MEA_rec_patch_ground_truth_cell7.raw.h5" (mode r)>

In [13]: h.keys()
Out[13]: <KeysViewHDF5 ['bits', 'mapping', 'message_0', 'proc0', 'settings', 'sig', 'time', 'version']>

In [14]: h['sig']
Out[14]: <HDF5 dataset "sig": shape (1028, 3723600), type "<u2">

In [15]: h['sig'][0]
^C
^C^C^C^C^C^C^C^C

---

KeyboardInterrupt Traceback (most recent call last)
Cell In[15], line 1
----> 1 h['sig'][0]

File h5py/\_objects.pyx:54, in h5py.\_objects.with_phil.wrapper()

File h5py/\_objects.pyx:55, in h5py.\_objects.with_phil.wrapper()

File ~/rtsort-web/venv/lib/python3.11/site-packages/h5py/\_hl/dataset.py:802, in Dataset.**getitem**(self, args, new_dtype)
800 if self.\_fast_read_ok and (new_dtype is None):
801 try:
--> 802 return self.\_fast_reader.read(args)
803 except TypeError:
804 pass # Fall back to Python read pathway below

KeyboardInterrupt:

In [16]:

In [16]: h['sig'][0][0]
Out[16]: np.uint16(513)

In [17]: h["settings"
...: ]
Out[17]: <HDF5 group "/settings" (3 members)>

In [18]: h["settings"].keys()
Out[18]: <KeysViewHDF5 ['gain', 'hpf', 'lsb']>

In [19]: h["settings"]["gain"]
Out[19]: <HDF5 dataset "gain": shape (1,), type "<f8">

## In [20]: h["settings"]["gain"].values

AttributeError Traceback (most recent call last)
Cell In[20], line 1
----> 1 h["settings"]["gain"].values

AttributeError: 'Dataset' object has no attribute 'values'

In [21]: h["settings"]["gain"][0]
Out[21]: np.float64(512.0)

In [22]: h['settings']['lsb'][0] \* 1e6
Out[22]: np.float64(6.29425)

In [23]: h['settings']['lsb'][0] _ 1e6 _ h['sig'][0][0]
Out[23]: np.float64(3228.95025)

In [24]:
