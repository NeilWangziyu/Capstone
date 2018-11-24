package com.example.administrator.gsontest;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.ImageView;
import android.widget.MediaController;
import android.widget.Toast;
import com.example.administrator.gsontest.MyService;

public class Static extends AppCompatActivity {
    private ImageView mImage;
    private IntentFilter mIntentFilter1;
    public static final String mBroadcastStringAction = "com.example.administrator.gsontest.string";
    public void onResume() {
        super.onResume();
        registerReceiver(mReceiver1, mIntentFilter1);

    }

    public void onDestroy() {
        super.onDestroy();
       /* unregisterReceiver(mReceiver1);*/
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_static);
        mImage = findViewById(R.id.image);
        mImage.setImageResource(R.drawable.staticges);
        mIntentFilter1 = new IntentFilter();
        mIntentFilter1.addAction(mBroadcastStringAction);
        Log.d("Static's ", "onCreate: is running ");
        registerReceiver(mReceiver1, mIntentFilter1);
        Intent serviceIntent = new Intent(this, MyService.class);
        startService(serviceIntent);

    }
    private BroadcastReceiver mReceiver1 = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            /*Intent startintent = new Intent(Static.this,MyService.class);
            startService(startintent);*/
            String gesnum = intent.getStringExtra("ges_num");
            System.out.println("Static接收到的广播数据：" + gesnum);
            if (intent.getAction().equals(mBroadcastStringAction)) {
                if (!gesnum.equals("10")) {
                    Toast.makeText(Static.this, "gesnum = " + gesnum + ",play the video", Toast.LENGTH_SHORT).show();
                    Intent jumpintent = new Intent(Static.this,video.class);
                    jumpintent.putExtra("ges_num",gesnum);

                    /*Intent stopIntent = new Intent(Static.this,
                            MyService.class);
                    stopService(stopIntent);*/
                    startActivity(jumpintent);



                }
                else {
                    Toast.makeText(Static.this, "waiting for the gesture", Toast.LENGTH_SHORT).show();

                }




            }

        }
    };
}
