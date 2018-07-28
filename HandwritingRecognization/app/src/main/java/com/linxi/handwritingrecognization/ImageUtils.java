package com.linxi.handwritingrecognization;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.util.Base64;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.StreamCorruptedException;
import java.util.List;

public class ImageUtils {
    public static Bitmap scaleImage(Bitmap bitmap, int width, int height)
    {
        Mat src = new Mat();
        Mat dst = new Mat();
        Utils.bitmapToMat(bitmap, src);
        //new Size(width, height)
        Imgproc.resize(src, dst, new Size(width,height),0,0,Imgproc.INTER_AREA);
        Bitmap bitmap1 = Bitmap.createBitmap(dst.cols(),dst.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(dst, bitmap1);
        return bitmap1;
    }

    public static Uri getUri(Context context, String path) {

        File picPath = new File(path);
        return Uri.fromFile(picPath);
    }

    public static String listToString(float[] list)throws IOException {
        // 实例化一个ByteArrayOutputStream对象，用来装载压缩后的字节文件。
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        // 然后将得到的字符数据装载到ObjectOutputStream
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(
                byteArrayOutputStream);
        // writeObject 方法负责写入特定类的对象的状态，以便相应的 readObject 方法可以还原它
        objectOutputStream.writeObject(list);
        // 最后，用Base64.encode将字节文件转换成Base64编码保存在String中
        String listString = new String(Base64.encode(
                byteArrayOutputStream.toByteArray(), Base64.DEFAULT));
        // 关闭objectOutputStream
        objectOutputStream.close();
        return listString;
    }

    @SuppressWarnings("unchecked")
    public static float[] StringToList(String listString) throws StreamCorruptedException, IOException,
            ClassNotFoundException {
        byte[] mobileBytes = Base64.decode(listString.getBytes(),
                Base64.DEFAULT);
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(
                mobileBytes);
        ObjectInputStream objectInputStream = new ObjectInputStream(
                byteArrayInputStream);
        float[] WeatherList = (float[]) objectInputStream
                .readObject();
        objectInputStream.close();
        return WeatherList;
    }
}
