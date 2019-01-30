import * as tf from '@tensorflow/tfjs';
import { Tensor } from '@tensorflow/tfjs';

export class DatasetController {
  public savedX: Tensor;
  public savedY: Tensor;

  constructor(private numClasses: number) {}

  addExample(example: Tensor, label: number) {
    const y = tf.tidy(() =>
      tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses)
    );

    if (this.savedX == null) {
      this.savedX = tf.keep(example);
      this.savedY = tf.keep(y);
    } else {
      this.savedX = tf.keep(this.savedX.concat(example, 0));
      this.savedY = tf.keep(this.savedY.concat(y, 0));

      y.dispose();
    }
  }
}
