package com.conectec.rn_movilapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ConnectorClassifier(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val modelPath = "mejor_modelo_ft.tflite"
    private val imageSize = 224 // Tamaño de imagen requerido
    private val numClasses = 20 // Número de clases en el modelo

    private val classNames = arrayOf(
        "Conector Lightning (Apple)",
        "Cable Audio Óptico",
        "Clavija US (Americana)",
        "Clavija US (Americana) 3 pines",
        "Cable Coaxial",
        "Adaptador de corriente 6 salidas",
        "Adaptador de corriente (clavija redonda)",
        "Conector DisplayPort",
        "Conector HDMI",
        "Adaptador jack de audio de 3.5mm",
        "Cargador magnetico",
        "Conector Micro HDMI",
        "Conector Micro-USB",
        "Adaptador de corriente multicontacto",
        "Conector RCA",
        "Conector RJ-45 (Ethernet)",
        "Adaptador multipuerto USB hub",
        "Conector USB tipo A",
        "Conector USB tipo C",
        "Conector VGA"
    )

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                // Configuración opcional para mejorar el rendimiento
                setNumThreads(4) // Usar 4 hilos para inferencia
            }
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            throw RuntimeException("Error al cargar el modelo TFLite", e)
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun classify(bitmap: Bitmap): ClassificationResult {
        // 1. Convertir la imagen a escala de grises
        val grayscaleBitmap = convertToGrayscale(bitmap)

        // 2. Preprocesamiento para el modelo
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Ajusta estos valores según cómo fue preprocesado tu dataset
            .build()

        // 3. Convertir a TensorImage y procesar
        var tensorImage = TensorImage.fromBitmap(grayscaleBitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // 4. Ejecutar inferencia
        val outputBuffer = Array(1) { FloatArray(numClasses) }
        interpreter?.run(tensorImage.buffer, outputBuffer)

        // 5. Obtener resultados
        val (maxIndex, maxProbability) = getMaxResult(outputBuffer[0])

        return ClassificationResult(
            className = classNames[maxIndex],
            confidence = maxProbability,
            allProbabilities = outputBuffer[0].toList()
        )
    }

    private fun convertToGrayscale(bitmap: Bitmap): Bitmap {
        // Crear un bitmap en escala de grises
        val grayscaleBitmap = Bitmap.createBitmap(
            bitmap.width,
            bitmap.height,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(grayscaleBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix().apply {
            setSaturation(0f) // 0 = escala de grises
        }

        paint.colorFilter = ColorMatrixColorFilter(colorMatrix)
        canvas.drawBitmap(bitmap, 0f, 0f, paint)

        return grayscaleBitmap
    }

    private fun getMaxResult(probabilities: FloatArray): Pair<Int, Float> {
        var maxIndex = 0
        var maxProbability = probabilities[0]

        for (i in 1 until probabilities.size) {
            if (probabilities[i] > maxProbability) {
                maxIndex = i
                maxProbability = probabilities[i]
            }
        }

        return Pair(maxIndex, maxProbability)
    }

    fun close() {
        interpreter?.close()
    }
}

data class ClassificationResult(
    val className: String,
    val confidence: Float,
    val allProbabilities: List<Float> = emptyList()
)