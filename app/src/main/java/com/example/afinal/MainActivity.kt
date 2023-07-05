package com.example.afinal

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.afinal.ml.MobilenetV110224Quant
import com.example.afinal.utils.Constants.CAMERA_REQUEST
import com.example.afinal.utils.Constants.SELECT_PICTURE_REQUEST
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var selectButton: Button
    private lateinit var predictButton: Button
    private lateinit var cameraButton: Button
    private lateinit var resultTextView: TextView
    private lateinit var imageView: ImageView
    private lateinit var imageBitmap: Bitmap
    private lateinit var labelList: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        setTheme(R.style.Base_Theme_FInal)
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        selectButton = findViewById(R.id.select_btn)
        predictButton = findViewById(R.id.predict_btn)
        cameraButton = findViewById(R.id.take_picture)
        resultTextView = findViewById(R.id.result_view)
        imageView = findViewById(R.id.image_view)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        labelList = application.assets.open("label.txt").bufferedReader().readLines()

        cameraButton.setOnClickListener {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(cameraIntent, CAMERA_REQUEST)
        }

        selectButton.setOnClickListener {
            val intent = Intent()
            intent.action = Intent.ACTION_GET_CONTENT
            intent.type = "image/*"
            startActivityForResult(intent, SELECT_PICTURE_REQUEST)
        }

        predictButton.setOnClickListener {
            Thread {
                val (prediction) = predict(imageProcessor, imageBitmap)
                Handler(Looper.getMainLooper()).post {
                    resultTextView.text = "i am pretty sure it is $prediction"
                }
            }.start()
        }
    }

    private fun predict(imageProcessor: ImageProcessor, bitmap: Bitmap): Pair<String, Float> {
        var tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        val model = MobilenetV110224Quant.newInstance(this)
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
        inputFeature0.loadBuffer(tensorImage.buffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

        var maxIndex = 0
        var maxConfidence = outputFeature0[0]
        outputFeature0.forEachIndexed { index, fl ->
            if (maxConfidence < fl) {
                maxConfidence = fl
                maxIndex = index
            }
        }
        model.close()
        return Pair(labelList[maxIndex], maxConfidence)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == SELECT_PICTURE_REQUEST) {
            val uri = data?.data
            imageBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imageView.setImageBitmap(imageBitmap)
        } else if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK) {
            imageBitmap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(imageBitmap)
        }
    }
}
