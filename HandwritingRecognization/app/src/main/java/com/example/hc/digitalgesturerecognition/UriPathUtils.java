package com.example.hc.digitalgesturerecognition;

import java.io.File;

import android.content.Context;
import android.net.Uri;

public class UriPathUtils {

    public static Uri getUri(Context context,String path) {

        File picPath = new File(path);
        Uri uri = null;
//        if(picPath.exists()) {
        uri = Uri.fromFile(picPath);
//        }

        return uri;
    }

}