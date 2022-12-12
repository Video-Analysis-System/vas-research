# MỘT SỐ CÔNG CỤ HỖ TRỢ  
## 1. Download ảnh tự động 
### 1.1 Down load với google image 
- Sử dụng mạng ngoài VTS. 
- Download ảnh từ chrome. 
- Github: 
https://github.com/ultralytics/google-images-download
### 1.2 Down load ảnh từ bing 
- Sử dụng được mạng VTS
- Down load ảnh sử dụng trình duyệt chrome, trang tìm kiếm bing.com 
- Chọn Bing, Chrome, Proxy: 10.61.11.42:3128 
- Github: https://github.com/sczhengyabin/Image-Downloader.git
## 2. Trích xuất và lưu ảnh từ video 
- Trích xuất các frame của video, lưu lại thành ảnh.
- StackOverFlow: https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
- Github: 
- Run code: 
```
python3 capture_video.py --pathIn video/video.mp4 --pathOut image --time 1000 --imageName frame
```
## 3. Xử lý dataset Yolo
### 3.1 Sửa đối tương file nhãn 
- Sửa đổi đốí tượng trong các file nhãn yolo.
- StackOverFlow: https://stackoverflow.com/questions/66511872/how-to-modify-the-value-of-a-yolo-txt-file-in-python
- Github : 
- Run code: 
```
python3 edit_object_label.py --pathLabels labels --number 0
```
### 3.2 Xóa ảnh không có nhãn trong thư mục
- Những ảnh sai, ảnh lỗi không có đối tượng được bỏ qua khi đánh nhãn nên chúng không có file nhãn. Để xóa chúng đi: 
- StackExchange: https://unix.stackexchange.com/questions/528490/python-removing-jpg-files-without-matching-txt-files
- Github: 
- Run code: 
```
python3 remove_image.py --pathLabels labels  --pathImages images 
```
### 3.3 Chia dataset 
- Chia images và labels thành train/test theo tỷ lệ tùy chọn.
- StackOverFlow: https://stackoverflow.com/questions/66579311/yolov4-custom-dataset-train-test-split
- Github: https://github.com/pylabel-project/samples/blob/main/dataset_splitting.ipynb
## 5. Đánh nhãn tự động bài toán object detection  
### 5.1 Sử dụng GLIPv2
- Điền keyword tên đối tượng => phát hiện đối tượng, đánh nhãn các file ảnh ghi ra file txt format yolo.
- Google colab: https://colab.research.google.com/drive/1849vX4RtGp7nXrbBcs0fXCeM30Ktwwmp
### 5.2 Sử dụng Yolo Labeler
- Xóa background ảnh và đánh nhãn theo format Yolo
- Chỉ sử dụng cho ảnh có một đối tượng duy nhất trong một ảnh.
- Pypi: https://pypi.org/project/yolo-labeler/
## 6. Dataset
### 6.1 Các trang đăng tải dataset
- Roboflow: https://universe.roboflow.com/
- Images.cv: https://images.cv/
- Kaggle: https://www.kaggle.com/datasets
- Mì AI: https://miai.vn/thu-vien-mi-ai/
### 6.2 Các bộ dataset lớn nổi tiếng
- OpenImage: 9M images, 20.638 categories. Link: https://storage.googleapis.com/openimages/web/download_v7.html
- Objects356: 2M images, 365 categories, 30 million bounding boxes. Link:    https://www.objects365.org/overview.html
- COCO: 330K images, 80 categories. Link: https://cocodataset.org/#home
