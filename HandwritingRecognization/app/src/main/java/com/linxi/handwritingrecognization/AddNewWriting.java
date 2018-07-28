package com.linxi.handwritingrecognization;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentUris;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;

import java.io.IOException;

import com.linxi.handwritingrecognization.SharedPreferenceUtils;

public class AddNewWriting extends AppCompatActivity implements View.OnClickListener {
    //加载OpenCV,必须先加载
    static{
        if(!OpenCVLoader.initDebug())
        {
            Log.d("OpenCV", "init failed");
        }
    }

    private Button btnChooseImg;
    private Button btnAddWriting;
    private EditText etWritingNote;
    private TextView tvShowMessage;
    private ImageView ivPicture;

    private static final int CHOOSE_PHOTO=2;
    private static final int REQUEST_PERMISSION=1;
    private float[] vectorResult=null;

    private Classifier classifier;
    private static final String MODEL_FILE="file:///android_asset/gesture.pb";
    private static final String VECTOR="WritingVector";
    private static final String WRITINGNOTE="WritingNote";

    private void init(){
        btnChooseImg=findViewById(R.id.btnChooseWriting);
        btnAddWriting=findViewById(R.id.btnAddWriting);
        etWritingNote=findViewById(R.id.etGestureNote);
        tvShowMessage=findViewById(R.id.tvShowMessage);
        ivPicture=findViewById(R.id.ivPicture);

        btnChooseImg.setOnClickListener(this);
        btnAddWriting.setOnClickListener(this);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_new_writing);

        init();
    }

    @SuppressLint("SetTextI18n")
    @Override
    public void onClick(View v) {
        switch (v.getId()){
            case R.id.btnChooseWriting:
                if (ContextCompat.checkSelfPermission(AddNewWriting.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED){
                    ActivityCompat.requestPermissions(AddNewWriting.this,new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},REQUEST_PERMISSION);
                }else {
                    openAlbum();
                }
                break;
            case R.id.btnAddWriting:
                addNewWriting();
                Toast.makeText(AddNewWriting.this,"add new handwriting success",Toast.LENGTH_SHORT).show();
                Intent intent=new Intent(AddNewWriting.this,MainActivity.class);
                startActivity(intent);
            default:
                break;
        }
    }

    private void addNewWriting() {
        if(vectorResult!=null&&etWritingNote.getText().toString().length()>0){
            String writingNote=etWritingNote.getText().toString();
            SharedPreferenceUtils.putBean(AddNewWriting.this,VECTOR,vectorResult);
            SharedPreferenceUtils.putBean(AddNewWriting.this,WRITINGNOTE,writingNote);
        }
        else{
            Toast.makeText(AddNewWriting.this,"all inputs are required",Toast.LENGTH_SHORT).show();
        }
    }

    public void openAlbum(){
        Intent intent = new Intent("android.intent.action.GET_CONTENT");
        intent.setType("image/*");
        startActivityForResult(intent,CHOOSE_PHOTO);//打开相册
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode){
            case 1:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED){
                    openAlbum();
                }else {
                    Toast.makeText(this,"You denied the permission",Toast.LENGTH_SHORT).show();
                }
                break;
            default:
                break;
        }
    }

    @SuppressLint("SetTextI18n")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode){
            case CHOOSE_PHOTO:
                try {
                    handleImage(data);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                break;
            default:
                break;
        }
    }

    private void handleImage(Intent data) throws IOException {
        String imagePath = null;
        Uri uri = data.getData();
        if (DocumentsContract.isDocumentUri(this,uri)){
            //如果是document类型的Uri,则通过document id处理
            String docId = DocumentsContract.getDocumentId(uri);
            assert uri != null;
            if ("com.android.providers.media.documents".equals(uri.getAuthority())){
                String id = docId.split(":")[1];//解析出数字格式的id
                String selection = MediaStore.Images.Media._ID + "=" + id;
                imagePath = getImagePath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,selection);
            }else if ("com.android.providers.downloads.documents".equals(uri.getAuthority())){
                Uri contentUri = ContentUris.withAppendedId(Uri.parse("content://downloads/public_downloads"),Long.valueOf(docId));
                imagePath = getImagePath(contentUri,null);
            }
        }else {
            assert uri != null;
            if ("content".equalsIgnoreCase(uri.getScheme())){
                //如果是content类型的Uri，则使用普通方式处理
                imagePath = getImagePath(uri,null);
            }else if ("file".equalsIgnoreCase(uri.getScheme())){
                //如果是file类型的Uri，直接获取图片路径即可
                imagePath = uri.getPath();
            }
        }
        if(imagePath!=null){
            Bitmap photo= BitmapFactory.decodeFile(imagePath);
            Bitmap bitmapResizedPhoto=ImageUtils.scaleImage(photo,64,64);

            classifier=new Classifier(getAssets(),MODEL_FILE);
            vectorResult=classifier.getVector(bitmapResizedPhoto);
            ivPicture.setImageBitmap(photo);
        }
    }

    private String getImagePath(Uri uri,String selection){
        String path = null;
        //通过Uri和selection来获取真实的图片路径
        Cursor cursor = getContentResolver().query(uri,null,selection,null,null);
        if (cursor != null){
            if (cursor.moveToFirst()){
                path = cursor.getString(cursor.getColumnIndex(MediaStore.Images.Media.DATA));
            }
            cursor.close();
        }
        return path;
    }
}
