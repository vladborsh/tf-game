import * as tf from '@tensorflow/tfjs';
import { Tensor4D, Rank, Tensor, Tensor3D } from '@tensorflow/tfjs';
import { Observable, Observer } from 'rxjs';

/**
 * A class that wraps webcam video elements to capture Tensor4Ds.
 */
export class WebCamera {

  constructor(private video: HTMLVideoElement) {}

  private static adjustVideoSize(video: HTMLVideoElement, width: number, height: number): void {
    const aspectRatio = width / height;
    if (width >= height) {
      video.width = aspectRatio * video.height;
    } else if (width < height) {
      video.height = video.width / aspectRatio;
    }
  }

  public setup(): Observable<void> {
    return Observable.create((observer: Observer<void>) => {
      navigator.getUserMedia = navigator.getUserMedia;

      if (!navigator.getUserMedia) {
        observer.error('UserMedia not found');
      }

      navigator.getUserMedia(
        { video: true },
        (stream: MediaStream) => {
          this.video.srcObject = stream;
          this.video.addEventListener('loadeddata', async () => {
            WebCamera.adjustVideoSize(this.video, this.video.videoWidth, this.video.videoHeight);
            observer.next(undefined);
            observer.complete();
          }, false);
        },
        observer.error,
      );

    });
  }

  public capture(): Tensor {
    return tf.tidy(() => {
      const croppedImage = this.cropImage(tf.fromPixels(this.video));
      const batchedImage = croppedImage.expandDims(0);

      return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
  }

  public cropImage(img: Tensor3D): Tensor<Rank.R3> {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - size / 2;
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - size / 2;
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  }
}
