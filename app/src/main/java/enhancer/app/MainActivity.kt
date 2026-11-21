package com.enhancer.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.media.*
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.View
import android.view.WindowManager
import android.widget.*
import java.io.File
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.roundToInt

class MainActivity : Activity() {
    private val RES = linkedMapOf("720p" to 1280, "1080p" to 1920, "2K" to 2560, "4K" to 3840)
    private val BR = mapOf("720p" to 8_000_000, "1080p" to 16_000_000, "2K" to 28_000_000, "4K" to 45_000_000)
    private var res = "1080p"
    private var nnapi = false
    private var smooth = false
    private lateinit var status: TextView
    private lateinit var bar: ProgressBar
    private lateinit var preview: ImageView
    private lateinit var open: Button
    private lateinit var bPhoto: Button
    private lateinit var bVideo: Button
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var esr: OrtSession? = null
    private var rife: OrtSession? = null
    private var lastUri: Uri? = null
    private var lastMime = "image/jpeg"
    private val ui = Handler(Looper.getMainLooper())
    @Volatile private var busy = false

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private fun say(t: String, p: Int = -1) = ui.post { status.text = t; if (p >= 0) bar.progress = p }

    private fun radios(opts: List<String>, sel: Int, on: (String) -> Unit): RadioGroup {
        val g = RadioGroup(this); g.orientation = RadioGroup.HORIZONTAL
        opts.forEach { t -> val b = RadioButton(this); b.text = t; b.id = View.generateViewId(); g.addView(b) }
        g.setOnCheckedChangeListener { gr, id -> on(gr.findViewById<RadioButton>(id).text.toString()) }
        (g.getChildAt(sel) as RadioButton).isChecked = true
        return g
    }

    override fun onCreate(s: Bundle?) {
        super.onCreate(s)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val col = LinearLayout(this); col.orientation = LinearLayout.VERTICAL; col.setPadding(dp(18), dp(28), dp(18), dp(28))
        fun label(t: String) = TextView(this).apply { text = t; textSize = 17f; setPadding(0, dp(16), 0, dp(4)); setTypeface(null, android.graphics.Typeface.BOLD) }
        col.addView(TextView(this).apply { text = "Enhancer"; textSize = 26f; gravity = android.view.Gravity.CENTER; setTypeface(null, android.graphics.Typeface.BOLD) })
        preview = ImageView(this); preview.adjustViewBounds = true; preview.maxHeight = dp(300); col.addView(preview)
        col.addView(label("Quality"))
        col.addView(radios(RES.keys.toList(), 1) { res = it })
        col.addView(label("Processor"))
        col.addView(radios(listOf("CPU", "GPU (experimental)"), 0) { nnapi = it.startsWith("GPU"); closeSessions() })
        val cb = CheckBox(this); cb.text = "Smooth video to 60fps (RIFE)"; cb.setOnCheckedChangeListener { _, c -> smooth = c }
        col.addView(cb)
        bPhoto = Button(this); bPhoto.text = "Choose photo"; bPhoto.setOnClickListener { pick("image/*", 1) }
        bVideo = Button(this); bVideo.text = "Choose video"; bVideo.setOnClickListener { pick("video/*", 2) }
        col.addView(bPhoto); col.addView(bVideo)
        bar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal); col.addView(bar)
        status = TextView(this); status.gravity = android.view.Gravity.CENTER; status.text = "Results are saved to Pictures/Enhancer and Movies/Enhancer."; col.addView(status)
        open = Button(this); open.text = "Open result"; open.visibility = View.GONE
        open.setOnClickListener {
            lastUri?.let { u -> startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(u, lastMime).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)) }
        }
        col.addView(open)
        setContentView(ScrollView(this).apply { addView(col) })
    }

    private fun pick(type: String, code: Int) {
        if (busy) return
        startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).setType(type).addCategory(Intent.CATEGORY_OPENABLE), code)
    }

    override fun onActivityResult(rc: Int, res2: Int, data: Intent?) {
        super.onActivityResult(rc, res2, data)
        val uri = data?.data ?: return
        if (res2 != RESULT_OK || busy) return
        busy = true; bPhoto.isEnabled = false; bVideo.isEnabled = false; open.visibility = View.GONE
        Thread {
            try {
                closeSessions()
                if (rc == 1) doPhoto(uri) else doVideo(uri)
            } catch (e: Throwable) {
                if (nnapi) say("GPU mode failed. Switch Processor to CPU and try again.") else say("Error: " + (e.message ?: e.javaClass.simpleName))
            } finally {
                busy = false
                ui.post { bPhoto.isEnabled = true; bVideo.isEnabled = true; if (lastUri != null) open.visibility = View.VISIBLE }
            }
        }.start()
    }

    private fun closeSessions() { try { esr?.close(); rife?.close() } catch (_: Throwable) {}; esr = null; rife = null }

    private fun session(asset: String): OrtSession {
        val o = OrtSession.SessionOptions()
        val cores = Runtime.getRuntime().availableProcessors()
        o.setIntraOpNumThreads(if (res == "720p") minOf(4, cores) else cores)
        if (nnapi) { try { o.addNnapi() } catch (_: Throwable) {} } else { try { o.addXnnpack(mapOf("intra_op_num_threads" to "4")) } catch (_: Throwable) {} }
        return env.createSession(assets.open(asset).readBytes(), o)
    }

    private fun c8(v: Float) = (v * 255f + .5f).toInt().coerceIn(0, 255)

    /** Real-ESRGAN x4, tiled so phones don't run out of memory. */
    private fun esrgan(src: Bitmap, prog: (Int) -> Unit = {}): Bitmap {
        val s = esr ?: session("realesrgan.onnx").also { esr = it }
        val W = src.width; val H = src.height
        val out = Bitmap.createBitmap(W * 4, H * 4, Bitmap.Config.ARGB_8888)
        val T = 96; val P = 8
        val nx = (W + T - 1) / T; val ny = (H + T - 1) / T; var n = 0
        for (ty in 0 until ny) for (tx in 0 until nx) {
            val x0 = tx * T; val y0 = ty * T; val x1 = minOf(W, x0 + T); val y1 = minOf(H, y0 + T)
            val ix0 = maxOf(0, x0 - P); val iy0 = maxOf(0, y0 - P); val ix1 = minOf(W, x1 + P); val iy1 = minOf(H, y1 + P)
            val w = ix1 - ix0; val h = iy1 - iy0
            val px = IntArray(w * h); src.getPixels(px, 0, w, ix0, iy0, w, h)
            val f = FloatArray(3 * w * h); val pl = w * h
            for (i in 0 until pl) { val c = px[i]; f[i] = ((c shr 16) and 255) / 255f; f[pl + i] = ((c shr 8) and 255) / 255f; f[2 * pl + i] = (c and 255) / 255f }
            val t = OnnxTensor.createTensor(env, FloatBuffer.wrap(f), longArrayOf(1, 3, h.toLong(), w.toLong()))
            val r = s.run(mapOf(s.inputNames.first() to t))
            val o = (r[0] as OnnxTensor).floatBuffer
            val ow = w * 4; val oh = h * 4; val op = ow * oh
            val cx = (x0 - ix0) * 4; val cy = (y0 - iy0) * 4; val cw = (x1 - x0) * 4; val ch = (y1 - y0) * 4
            val tp = IntArray(cw * ch)
            for (yy in 0 until ch) for (xx in 0 until cw) {
                val idx = (cy + yy) * ow + cx + xx
                tp[yy * cw + xx] = (0xFF shl 24) or (c8(o.get(idx)) shl 16) or (c8(o.get(op + idx)) shl 8) or c8(o.get(2 * op + idx))
            }
            out.setPixels(tp, 0, cw, x0 * 4, y0 * 4, cw, ch)
            r.close(); t.close()
            n++; prog(n * 100 / (nx * ny))
        }
        return out
    }

    /** RIFE: the frame halfway between a and b. */
    private fun rifeMid(a: Bitmap, b: Bitmap): Bitmap {
        val s = rife ?: session("rife.onnx").also { rife = it }
        val W = a.width; val H = a.height; val pw = (W + 63) / 64 * 64; val ph = (H + 63) / 64 * 64
        fun tens(bm: Bitmap): OnnxTensor {
            val px = IntArray(W * H); bm.getPixels(px, 0, W, 0, 0, W, H)
            val f = FloatArray(3 * pw * ph); val pl = pw * ph
            for (y in 0 until H) for (x in 0 until W) {
                val c = px[y * W + x]; val i = y * pw + x
                f[i] = ((c shr 16) and 255) / 255f; f[pl + i] = ((c shr 8) and 255) / 255f; f[2 * pl + i] = (c and 255) / 255f
            }
            return OnnxTensor.createTensor(env, FloatBuffer.wrap(f), longArrayOf(1, 3, ph.toLong(), pw.toLong()))
        }
        val ta = tens(a); val tb = tens(b)
        val ts = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(.5f)), longArrayOf(1))
        val r = s.run(mapOf("img0" to ta, "img1" to tb, "timestep" to ts))
        val o = (r[0] as OnnxTensor).floatBuffer; val pl = pw * ph
        val px = IntArray(W * H)
        for (y in 0 until H) for (x in 0 until W) {
            val i = y * pw + x
            px[y * W + x] = (0xFF shl 24) or (c8(o.get(i)) shl 16) or (c8(o.get(pl + i)) shl 8) or c8(o.get(2 * pl + i))
        }
        r.close(); ta.close(); tb.close(); ts.close()
        return Bitmap.createBitmap(px, W, H, Bitmap.Config.ARGB_8888)
    }

    private fun save(name: String, mime: String, write: (OutputStream) -> Unit): Uri {
        val vid = mime.startsWith("video")
        val cv = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name); put(MediaStore.MediaColumns.MIME_TYPE, mime)
            put(MediaStore.MediaColumns.RELATIVE_PATH, (if (vid) Environment.DIRECTORY_MOVIES else Environment.DIRECTORY_PICTURES) + "/Enhancer")
        }
        val u = contentResolver.insert(if (vid) MediaStore.Video.Media.EXTERNAL_CONTENT_URI else MediaStore.Images.Media.EXTERNAL_CONTENT_URI, cv)!!
        contentResolver.openOutputStream(u)!!.use(write)
        lastUri = u; lastMime = mime
        return u
    }

    private fun showPreview(b: Bitmap) {
        val w = 900; val h = maxOf(1, b.height * w / b.width)
        val t = Bitmap.createScaledBitmap(b, w, h, true); ui.post { preview.setImageBitmap(t) }
    }

    private fun doPhoto(uri: Uri) {
        val src = ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { d, _, _ -> d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE }
        val target = RES[res]!!; val long = maxOf(src.width, src.height)
        val inLong = minOf(long, (target + 3) / 4); val sc = inLong.toFloat() / long
        val inp = if (sc < 1f) Bitmap.createScaledBitmap(src, maxOf(16, (src.width * sc).roundToInt()), maxOf(16, (src.height * sc).roundToInt()), true) else src
        say("Enhancing photo with Real-ESRGAN…", 0)
        val big = esrgan(inp) { say("Enhancing photo with Real-ESRGAN… $it%", it) }
        val f = target.toFloat() / maxOf(big.width, big.height)
        val outB = if (abs(f - 1f) > .01f) Bitmap.createScaledBitmap(big, (big.width * f).roundToInt(), (big.height * f).roundToInt(), true) else big
        save("Enhancer_${res}_${System.currentTimeMillis() / 1000}.jpg", "image/jpeg") { outB.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        showPreview(outB)
        say("Saved ${outB.width}×${outB.height} ($res) to Pictures/Enhancer.", 100)
    }

    private fun a16(x: Double) = maxOf(16, ((x + 8) / 16).toInt() * 16)
    private fun snapFps(x: Double) = listOf(24, 25, 30, 50, 60).minByOrNull { abs(it - x) }!!

    private fun nv12(bm: Bitmap): ByteArray {
        val w = bm.width; val h = bm.height; val px = IntArray(w * h); bm.getPixels(px, 0, w, 0, 0, w, h)
        val out = ByteArray(w * h * 3 / 2); var yi = 0; var ui = w * h
        for (y in 0 until h) for (x in 0 until w) {
            val c = px[y * w + x]; val r = (c shr 16) and 255; val g = (c shr 8) and 255; val b = c and 255
            out[yi++] = (((66 * r + 129 * g + 25 * b + 128) shr 8) + 16).toByte()
            if (y % 2 == 0 && x % 2 == 0) {
                out[ui++] = (((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128).toByte()
                out[ui++] = (((112 * r - 94 * g - 18 * b + 128) shr 8) + 128).toByte()
            }
        }
        return out
    }

    private fun doVideo(uri: Uri) {
        val mr = MediaMetadataRetriever(); mr.setDataSource(this, uri)
        fun md(k: Int) = mr.extractMetadata(k)?.toDoubleOrNull() ?: 0.0
        var w0 = md(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH); var h0 = md(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
        val rot = md(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION).toInt()
        if (rot == 90 || rot == 270) { val t = w0; w0 = h0; h0 = t }
        val dur = md(MediaMetadataRetriever.METADATA_KEY_DURATION); val fc = md(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)
        val srcFps = if (fc > 0 && dur > 0) fc * 1000.0 / dur else 30.0
        val doSmooth = smooth && srcFps <= 31
        val outFps = if (doSmooth) snapFps(srcFps * 2) else snapFps(srcFps)
        val target = RES[res]!!; val long = maxOf(w0, h0)
        val outW = a16(w0 * target / long); val outH = a16(h0 * target / long)
        val inLong = minOf(long, ((target + 3) / 4).toDouble()); val inW = maxOf(16, (w0 * inLong / long).toInt()); val inH = maxOf(16, (h0 * inLong / long).toInt())
        val total = (dur / 1000.0 * srcFps).toInt().coerceAtLeast(1)

        val ex = MediaExtractor(); ex.setDataSource(this, uri, null)
        var ai = -1; var afmt: MediaFormat? = null
        for (i in 0 until ex.trackCount) { val f = ex.getTrackFormat(i); if (ai < 0 && (f.getString(MediaFormat.KEY_MIME) ?: "").startsWith("audio")) { ai = i; afmt = f } }

        val tmp = File(cacheDir, "out.mp4"); tmp.delete()
        val mux = MediaMuxer(tmp.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val fmt = MediaFormat.createVideoFormat("video/avc", outW, outH)
        fmt.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar)
        fmt.setInteger(MediaFormat.KEY_BIT_RATE, BR[res]!!); fmt.setInteger(MediaFormat.KEY_FRAME_RATE, outFps); fmt.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        val enc = MediaCodec.createEncoderByType("video/avc"); enc.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE); enc.start()
        var vt = -1; var at = -1; var started = false

        fun drain(eos: Boolean) {
            val info = MediaCodec.BufferInfo()
            while (true) {
                val i = enc.dequeueOutputBuffer(info, if (eos) 10000L else 0L)
                if (i == MediaCodec.INFO_TRY_AGAIN_LATER) { if (!eos) return }
                else if (i == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    vt = mux.addTrack(enc.outputFormat)
                    if (afmt != null) { try { at = mux.addTrack(afmt) } catch (_: Throwable) { at = -1 } }
                    mux.start(); started = true
                } else if (i >= 0) {
                    val buf = enc.getOutputBuffer(i)!!
                    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                    if (info.size > 0 && started) mux.writeSampleData(vt, buf, info)
                    enc.releaseOutputBuffer(i, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }

        var k = 0L; val t0 = System.currentTimeMillis(); var lastShown: Bitmap? = null
        fun emit(bm: Bitmap) {
            val big = esrgan(bm); val sc = Bitmap.createScaledBitmap(big, outW, outH, true)
            val data = nv12(sc)
            var idx: Int
            while (true) { idx = enc.dequeueInputBuffer(10000); if (idx >= 0) break; drain(false) }
            val ib = enc.getInputBuffer(idx)!!; ib.clear(); ib.put(data)
            enc.queueInputBuffer(idx, 0, data.size, k * 1_000_000L / outFps, 0); k++
            drain(false); lastShown = sc
        }

        var prev: Bitmap? = null
        for (i in 0 until total) {
            val cur = mr.getScaledFrameAtTime((i * 1_000_000.0 / srcFps).toLong(), MediaMetadataRetriever.OPTION_CLOSEST, inW, inH) ?: prev ?: continue
            if (doSmooth && prev != null) emit(rifeMid(prev, cur))
            emit(cur); prev = cur
            val done = (i + 1) * 100 / total; val el = (System.currentTimeMillis() - t0) / 1000
            val eta = if (i > 0) el * (total - i - 1) / (i + 1) else 0
            say("Enhancing video $res/${outFps}fps… $done%  (about ${eta / 60}m ${eta % 60}s left)", done)
        }
        val idx = enc.dequeueInputBuffer(-1); enc.queueInputBuffer(idx, 0, 0, k * 1_000_000L / outFps, MediaCodec.BUFFER_FLAG_END_OF_STREAM); drain(true)
        if (afmt != null && at >= 0) {
            ex.selectTrack(ai); val ab = ByteBuffer.allocate(1 shl 20); val bi = MediaCodec.BufferInfo()
            while (true) { val n = ex.readSampleData(ab, 0); if (n < 0) break; bi.set(0, n, ex.sampleTime, ex.sampleFlags); mux.writeSampleData(at, ab, bi); ex.advance() }
        }
        mux.stop(); mux.release(); enc.stop(); enc.release(); ex.release(); mr.release()
        save("Enhancer_${res}_${outFps}fps_${System.currentTimeMillis() / 1000}.mp4", "video/mp4") { o -> tmp.inputStream().use { it.copyTo(o) } }
        tmp.delete(); lastShown?.let { showPreview(it) }
        say("Saved ${outW}×${outH} $res/${outFps}fps to Movies/Enhancer.", 100)
    }
}
