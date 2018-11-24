package com.example.administrator.gsontest;

import android.app.Service;
import android.content.ComponentName;
import android.content.Intent;
import android.os.Binder;
import android.os.IBinder;
import android.util.Log;


import android.widget.TextView;

import com.google.gson.Gson;

import java.util.Timer;
import java.util.TimerTask;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

import com.example.administrator.gsontest.video;

public class MyService extends Service {
    String ges1="";
    TextView responseText;
    public MyService() {

    }


    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        Log.d("MyService", "onCreate executed");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {

        Log.d("MyService", "onStartCommand executed");

            //String num = startConnection();
        /*Timer timer = new Timer();
        TimerTask task = new TimerTask() {
            public void run() {*/
                new Thread(new Runnable() {
                    @Override
                    public void run() {


                        do {
                            sendRequestWithOkHttp();
                            do {
                                try {
                                    Thread.sleep(100);
                                } catch (InterruptedException e) {
                                    e.printStackTrace();
                                }
                            } while (ges1.equals(""));
                            try {
                                Thread.sleep(200);
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                            Intent broadcastIntent2 = new Intent();
                            broadcastIntent2.setAction(Static.mBroadcastStringAction);
                            broadcastIntent2.putExtra("ges_num", ges1);
                            sendBroadcast(broadcastIntent2);
                            Log.d("MyStaticBroadcast", "is running!");
                        } while (ges1.equals("10"));

                /*Intent intent = new Intent("com.example.administrator.gsontest.MyReceiver");
                intent.setComponent(new ComponentName("com.example.administrator.gsontest","com.example.administrator.gsontest.MyReceiver"));*/
                        /*Intent broadcastIntent = new Intent();
                        broadcastIntent.setAction(video.mBroadcastStringAction);
                        broadcastIntent.putExtra("ges_num", ges1);
                        sendBroadcast(broadcastIntent);
                        Log.d("MyvideoBroadcast", "is running!");*/
                        /*try {
                            Thread.sleep(500);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }*/

                        //return ges1;
                    }

                }).start();



            return START_NOT_STICKY;
        }








    @Override
    public void onDestroy() {
        super.onDestroy();
    }
    public void parseJSONWithGSON(String jsonData) {
        Gson gson = new Gson();
        Ges1 gesture = gson.fromJson(jsonData, Ges1.class);
        ges1 = gesture.getResult();
        /*Log.d("MainActivity","gesture is" + gesture.getResult());*/
    }
    private void sendRequestWithOkHttp(){
        /*new Thread(new Runnable() {
            @Override
            public void run() {*/

                    /*try {
                        Thread.sleep(999); //wait for 1.5s
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }*/
                    try {
                        OkHttpClient client = new OkHttpClient();
                        Request request = new Request.Builder() //create new instance
                                .url("http://119.23.243.57:80/")
                                /*.url("http://www.baidu.com/")*/
                                .build();
                        Response response = client.newCall(request).execute();//create new instance
                        String responseData = response.body().string();
                        parseJSONWithGSON(responseData);// convert Gson to string
/*
                        showResponse(ges1);// set text in view
*/


                    } catch (Exception e) {
                        e.printStackTrace();
                    }


            /*}
        }).start();*/
    }

/*
    private void showResponse(final String response){
                responseText.setText(response);
            }
*/




}
