import 'dart:ffi';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:frontend/main.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:image_picker/image_picker.dart';

void main() async {
  runApp(MyAppCapture());
}

class MyAppCapture extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: ImageCapture(),
    );
  }
}

class ImageCapture extends StatefulWidget {
  createState() => _ImageCaptureState();
}

class _ImageCaptureState extends State<ImageCapture> {
  File _imageFile;

  Future<void> _cropImage() async {
    File cropped = await ImageCropper.cropImage(sourcePath: _imageFile.path);

    setState(() => {_imageFile = cropped ?? _imageFile});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      bottomNavigationBar: BottomAppBar(
        child: Row(children: <Widget>[
          IconButton(icon: Icon(Icons.photo_camera), onPressed: null),
          IconButton(icon: Icon(Icons.photo_library), onPressed: null)
        ]),
      ),
      body: ListView(
        children: <Widget>[],
      ),
    );
  }
}
