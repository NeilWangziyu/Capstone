package com.example.administrator.gsontest;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.media.MediaPlayer;
import android.net.Uri;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.MediaController;
import android.widget.Toast;
import android.widget.VideoView;


public class video extends AppCompatActivity {
    private VideoView mVideoView;
    private IntentFilter mIntentFilter;
    public static final String mBroadcastStringAction = "com.example.administrator.gsontest.string";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_video);
        mVideoView = findViewById(R.id.video);
        mIntentFilter = new IntentFilter();
        mIntentFilter.addAction(mBroadcastStringAction);
        /*Intent serviceIntent = new Intent(this, MyService.class);
        startService(serviceIntent);*/

   /* public void onResume() {
        super.onResume();
        registerReceiver(mReceiver, mIntentFilter);
    }

    public void onDestroy() {
        super.onDestroy();
        *//*unregisterReceiver(mReceiver)*//*;
    }*/


            Intent getIntent = getIntent();
            String gesnum = getIntent.getStringExtra("ges_num");
            System.out.println("接收到的数据：" + gesnum);
            MediaController mc = new MediaController(video.this);       // 创建一个MediaController对象
            mVideoView.setMediaController(mc);       // 将VideoView与MediaController关联起来
                switch (gesnum){
                    case "0":
                        mVideoView.setVideoURI(Uri.parse("android.resource://com.example.administrator.gsontest/" + R.raw.ges_zero));
                        break;
                    case "1":
                        mVideoView.setVideoURI(Uri.parse("android.resource://com.example.administrator.gsontest/" + R.raw.ges_first));
                        break;
                    case "2":
                        mVideoView.setVideoURI(Uri.parse("android.resource://com.example.administrator.gsontest/" + R.raw.ges_second));
                        break;
                    case "3":
                        mVideoView.setVideoURI(Uri.parse("android.resource://com.example.administrator.gsontest/" + R.raw.ges_third));
                        break;
                    case "4":
                        mVideoView.setVideoURI(Uri.parse("android.resource://com.example.administrator.gsontest/" + R.raw.ges_forth));
                        break;
                    case "5":
                        mVideoView.setVideoURI(Uri.parse("android.resource://com.example.administrator.gsontest/" + R.raw.ges_fifth));
                        break;
                    case "6":
                        mVideoView.setVideoURI(Uri.parse("android.resource://com.example.administrator.gsontest/" + R.raw.ges_sixth));
                        break;
                    case "7":
                        mVideoView.setVideoURI(Uri.parse("android.resource://com.example.administrator.gsontest/" + R.raw.ges_seventh));
                        break;
                    case "8":
                        mVideoView.setVideoURI(Uri.parse("android.resource://com.example.administrator.gsontest/" + R.raw.ges_eighth));
                        break;
                    case "9":
                        mVideoView.setVideoURI(Uri.parse("android.resource://com.example.administrator.gsontest/" + R.raw.ges_ninth));
                        break;
                    case "10":
                        /*Toast.makeText(video.this, "Waiting for gesture..gesnum = " + gesnum, Toast.LENGTH_SHORT).show();*/
                        Intent jumpintent = new Intent(video.this,Static.class);
                        Intent startintent = new Intent(video.this,MyService.class);
                        startService(startintent);
                        startActivity(jumpintent);
                    default:
                        break;
                }
                mVideoView.requestFocus();       // 设置VideoView获取焦点

                    mVideoView.start();



                /*mVideoView.start();*/

                Toast.makeText(video.this, "gesnum = " + gesnum, Toast.LENGTH_SHORT).show();
                mVideoView.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                    @Override
                    public void onCompletion(MediaPlayer mp) {

                        Intent intent = new Intent(video.this,Static.class);

                        startActivity(intent);

                        /*if (!gesnum.equals("10")) {
                            Toast.makeText(video.this, "gesnum = " + gesnum, Toast.LENGTH_SHORT).show();

                        }
                        else {
                            Toast.makeText(video.this, "Waiting for gesture..gesnum = " + gesnum, Toast.LENGTH_SHORT).show();
                        }*/


                        /*finish();*/
                    }
                });


        }
    }













/*
        try {
            video.start();      // 播放视频
            video.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
                @Override
                public void onCompletion(MediaPlayer mp) {
                    Intent intent = new Intent(video.this,Static.class);
                    startActivity(intent);

                }
            });
        }catch(Exception e) {
            e.printStackTrace();
        }
*/

        // 设置VideoView的Completion事件监听器
        /*video.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mp) {

            }
        });*/


