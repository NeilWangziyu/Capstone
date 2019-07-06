package com.example.administrator.gsontest;

import android.content.ComponentName;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.IBinder;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.google.gson.Gson;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import com.example.administrator.gsontest.video;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    TextView responseText;

 /*   private MyService.ConnectionBinder connectionBinder;
    private ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder service) {
            connectionBinder = (MyService.ConnectionBinder) service;
            connectionBinder.startConnection();
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {

        }
    };//bind service

*//*    Button sendRequest;*/

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button sendRequest = findViewById(R.id.send_request);
        /*Button test1 = findViewById(R.id.test1);*/
        responseText = findViewById(R.id.response_text);
        sendRequest.setOnClickListener(this);
        /*test1.setOnClickListener(this);*/
        }

        /*Thread.currentThread().sleep(15000);*//*等待15s*/



    @Override
    public void onClick(View v){
        switch (v.getId()){
            case R.id.send_request:
/*                sendRequestWithOkHttp();*/
                Intent intent = new Intent(MainActivity.this,Static.class);
                /*Intent startIntent = new Intent(this, MyService.class);
                startService(startIntent);//start the service*/
                startActivity(intent);
                /*Intent serviceIntent = new Intent(this, MyService.class);
                startService(serviceIntent);*/
/*
                intent.putExtra("ges_num", ges1);
*/
                break;

            /*case R.id.test1:
                Intent intent1 = new Intent(MainActivity.this,Static.class);
                Intent startIntent1 = new Intent(this, MyService.class);

                startService(startIntent1);
                startActivity(intent1);
                break;*/
            default:
                break;
        }
        /*if (v.getId() == R.id.send_request){
            sendRequestWithOkHttp();
            //跳转到视频播放界面
*//*            try {
                Thread.currentThread().sleep(1000);*//**//*等待1s*//**//*
            } catch (InterruptedException e) {
                e.printStackTrace();
            }*//*
            Intent intent = new Intent(MainActivity.this,video.class);
            startActivity(intent);
        if (v.getId() == R.id.test1){
            Intent intent1 = new Intent(MainActivity.this,Static.class);
            startActivity(intent1);
        }*/
/*            try {
                Thread.currentThread().sleep(10000);
                if (Vstate > 0 ){
                    Intent intent1 = new Intent(MainActivity.this,Static.class);
                    startActivity(intent1);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }*/



        }


    /*private void sendRequestWithOkHttp(){
        new Thread(new Runnable() {
            @Override
            public void run() {
                do {
                    try {
                        Thread.currentThread().sleep(999); //wait for 1.5s
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    try {
                        OkHttpClient client = new OkHttpClient();
                        Request request = new Request.Builder() //create new instance
                                .url("http://119.23.243.57:80/")
                                *//*.url("http://www.baidu.com/")*//*
                                .build();
                        Response response = client.newCall(request).execute();//create new instance
                        String responseData = response.body().string();
                        parseJSONWithGSON(responseData);// convert Gson to string
                        showResponse(ges1);// set text in view


                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                } while (true);
                *//*try {
                    OkHttpClient client = new OkHttpClient();
                    Request request = new Request.Builder()
                            .url("http://119.23.243.57:80/")
                            *//**//*.url("http://www.baidu.com/")*//**//*
                            .build();
                    Response response = client.newCall(request).execute();
                    String responseData = response.body().string();
                    parseJSONWithGSON(responseData);
                    showResponse(ges1);



                    
                } catch (Exception e) {
                    e.printStackTrace();
                }*//*
            }
        }).start();
    }
    public void parseJSONWithGSON(String jsonData) {
        Gson gson = new Gson();
        Ges1 gesture = gson.fromJson(jsonData, Ges1.class);
        ges1 = gesture.getResult();
        *//*Log.d("MainActivity","gesture is" + gesture.getResult());*//*
    }
    private void showResponse(final String response){
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                responseText.setText(response);
            }
        });
    }*/
}
