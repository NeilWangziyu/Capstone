package com.example.administrator.gsontest;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.widget.Toast;

public class MyReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {

        Toast.makeText(context, "received in receiver!", Toast.LENGTH_SHORT).show();
        if (intent.getAction().equals("com.example.administrator.gsontest.MyReceiver")) {
           String str = intent.getStringExtra("gesnum");
           System.out.println("接收到的广播的数据：" + str); }



    }
}
