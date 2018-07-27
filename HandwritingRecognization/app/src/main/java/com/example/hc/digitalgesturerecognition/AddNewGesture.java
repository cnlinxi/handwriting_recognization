package com.example.hc.digitalgesturerecognition;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import java.io.File;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.hc.digitalgesturerecognition.ActionSheetDialog.OnSheetItemClickListener;
import com.example.hc.digitalgesturerecognition.ActionSheetDialog.SheetItemColor;

import com.example.hc.digitalgesturerecognition.PhotoUtils;

public class AddNewGesture extends AppCompatActivity implements OnClickListener{

    private Button btnChooseGesture;
    private EditText etGestureNote;
    private Button btnAddGesture;

    private static final String TAG = CameraActivity.class.getSimpleName();

    private ImageView smallImg;
    private ImageView clipImg;
    private TextView tv1,tv2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {


        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_new_gesture);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

//        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
//        fab.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
//                        .setAction("Action", null).show();
//            }
//        });

        initView();
    }

    private void initView(){
        btnChooseGesture=(Button)findViewById(R.id.btnChooseGesture);
        btnAddGesture=(Button)findViewById(R.id.btnAddGesture);

        smallImg = (ImageView) findViewById(R.id.small_img);
        clipImg = (ImageView) findViewById(R.id.clip_pic);
        tv1 = (TextView) findViewById(R.id.small_img_title);
        tv2 = (TextView) findViewById(R.id.clip_pic_title);
    }

    @Override
    public void onClick(View v) {
        if(v.getId()==R.id.btnChooseGesture){
           photoOptions();
        }
    }

    private String path;

    @SuppressLint("SetTextI18n")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == PhotoUtils.NONE)
            return;
        // 拍照
        if (requestCode == PhotoUtils.PHOTOGRAPH) {
            // 设置文件保存路径这里放在跟目录下
            File picture = null;
            if (Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState())) {
                picture = new File(Environment.getExternalStorageDirectory() + PhotoUtils.imageName);
                if (!picture.exists()) {
                    picture = new File(Environment.getExternalStorageDirectory() + PhotoUtils.imageName);
                }
            } else {
                picture = new File(this.getFilesDir() + PhotoUtils.imageName);
                if (!picture.exists()) {
                    picture = new File(AddNewGesture.this.getFilesDir() + PhotoUtils.imageName);
                }
            }

            path = PhotoUtils.getPath(this);// 生成一个地址用于存放剪辑后的图片
            if (TextUtils.isEmpty(path)) {
                Log.e(TAG, "随机生成的用于存放剪辑后的图片的地址失败");
                return;
            }
            Uri imageUri = UriPathUtils.getUri(this, path);
            PhotoUtils.startPhotoZoom(AddNewGesture.this, Uri.fromFile(picture), PhotoUtils.PICTURE_HEIGHT, PhotoUtils.PICTURE_WIDTH, imageUri);
        }

        if (data == null)
            return;

        // 读取相册缩放图片
        if (requestCode == PhotoUtils.PHOTOZOOM) {

            path = PhotoUtils.getPath(this);// 生成一个地址用于存放剪辑后的图片
            if (TextUtils.isEmpty(path)) {
                Log.e(TAG, "随机生成的用于存放剪辑后的图片的地址失败");
                return;
            }
            Uri imageUri = UriPathUtils.getUri(this, path);
            PhotoUtils.startPhotoZoom(AddNewGesture.this, data.getData(), PhotoUtils.PICTURE_HEIGHT, PhotoUtils.PICTURE_WIDTH, imageUri);
        }
        // 处理结果
        if (requestCode == PhotoUtils.PHOTORESOULT) {
            /**
             * 在这里处理剪辑结果，可以获取缩略图，获取剪辑图片的地址。得到这些信息可以选则用于上传图片等等操作
             * */

            /**
             * 如，根据path获取剪辑后的图片
             */
            Bitmap bitmap = PhotoUtils.convertToBitmap(path,PhotoUtils.PICTURE_HEIGHT, PhotoUtils.PICTURE_WIDTH);
            if(bitmap != null){
                tv2.setText(bitmap.getHeight()+"x"+bitmap.getWidth()+"图");
                clipImg.setImageBitmap(bitmap);
            }

            Bitmap bitmap2 = PhotoUtils.convertToBitmap(path,120, 120);
            if(bitmap2 != null){
                tv1.setText(bitmap2.getHeight()+"x"+bitmap2.getWidth()+"图");
                smallImg.setImageBitmap(bitmap2);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    protected void photoOptions() {
        ActionSheetDialog mDialog = new ActionSheetDialog(this).builder();
        mDialog.setTitle("选择");
        mDialog.setCancelable(false);
        mDialog.addSheetItem("拍照", SheetItemColor.Blue, new OnSheetItemClickListener() {
            @Override
            public void onClick(int which) {
                PhotoUtils.photograph(AddNewGesture.this);
            }
        }).addSheetItem("从相册选取", SheetItemColor.Blue, new OnSheetItemClickListener() {
            @Override
            public void onClick(int which) {
                PhotoUtils.selectPictureFromAlbum(AddNewGesture.this);
            }
        }).show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }
}
