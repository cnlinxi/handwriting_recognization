package com.linxi.handwritingrecognization;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentUris;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;

import java.io.IOException;

public class MainActivity extends AppCompatActivity implements View.OnClickListener{
    static {
        if(!OpenCVLoader.initDebug())
        {
            Log.d("opencv","初始化失败");
        }
    }

    private Button btnAddPicture;
    private Button btnCheckWriting;
    private TextView tvResult;

    private static final int REQUEST_PERMISSION=1;
    private static final int CHOOSE_PHOTO=2;

    private Classifier classifier;
    private static final String MODEL_FILE="file:///android_asset/gesture.pb";

    private static final String VECTOR="WritingVector";
    private static final String WRITINGNOTE="WritingNote";

    private static Bitmap resizedImage;

    private void init() {
        btnAddPicture=findViewById(R.id.btnAddPicture);
        btnCheckWriting=findViewById(R.id.btnCheckWriting);
        tvResult=(TextView) findViewById(R.id.tvResult);

        btnAddPicture.setOnClickListener(this);
        btnCheckWriting.setOnClickListener(this);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent=new Intent(MainActivity.this,AddNewWriting.class);
                startActivity(intent);
            }
        });

        init();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }


    @Override
    public void onClick(View v) {
        switch (v.getId()){
            case R.id.btnAddPicture:
                if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED){
                    ActivityCompat.requestPermissions(MainActivity.this,new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},REQUEST_PERMISSION);
                }else {
                    openAlbum();
                }
                break;
            case R.id.btnCheckWriting:
                if(resizedImage!=null){
                    checkWriting();
                }
                break;
            default:
                break;
        }
    }

    @SuppressLint("DefaultLocale")
    private void checkWriting() {
        float[] originVector=(float[]) SharedPreferenceUtils.getBean(MainActivity.this,VECTOR);
        if(originVector==null){
            Toast.makeText(MainActivity.this,"please initialize...",Toast.LENGTH_SHORT).show();
            return;
        }
        classifier=new Classifier(getAssets(),MODEL_FILE);
        String writingNote=(String)SharedPreferenceUtils.getBean(MainActivity.this,WRITINGNOTE);
        float[] vector=classifier.getVector(resizedImage);
        double likelook=classifier.cosine(vector,originVector);
        tvResult.setText(String.format("the similarity of handwriting with %s is: %6f",writingNote,likelook));
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
        if(data==null){
            return;
        }
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
            resizedImage=ImageUtils.scaleImage(photo,64,64);
            Bitmap bitmapShowPhoto=ImageUtils.scaleImage(photo,150,150);
            Drawable drawable=new BitmapDrawable(bitmapShowPhoto);
            btnAddPicture.setBackground(drawable);
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
