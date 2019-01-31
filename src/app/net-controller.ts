import * as tf from '@tensorflow/tfjs';
import { Model, Rank, Tensor } from '@tensorflow/tfjs';
import { WebCamera } from './webcamera';
import { DatasetController } from './dataset-controller';
import { Config } from './config';
import { BehaviorSubject, Observable } from 'rxjs';

export class NetController {

  public webcam: WebCamera;

  private datasetController = new DatasetController(Config.NUM_CLASSES);
  private truncatedMobileNet: Model;
  private model: Model;

  constructor(private video: HTMLVideoElement) {
    this.webcam = new WebCamera(video);
    WebCamera.setup(video).then(() => {
      this.loadTruncatedMobileNet().then(val => this.truncatedMobileNet = val);
      this.init();
    });
  }

  public async init(): Promise<void> {
    this.truncatedMobileNet = await this.loadTruncatedMobileNet();
    tf.tidy(() => this.truncatedMobileNet.predict(this.webcam.capture()));
  }

  public async loadTruncatedMobileNet(): Promise<Model> {
    const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
    );

    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
  }

  public train$(): Observable<number> {
    return Observable.create(observer => {
      if (this.datasetController.savedX == null) {
        throw new Error('Add some examples before training!');
      }

      this.model = this.getModel();

      const optimizer = tf.train.adam(Config.LEARNING_RATE);
      this.model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

      const batchSize = Math.floor(this.datasetController.savedX.shape[0] * 0.4);
      if (!(batchSize > 0)) {
        throw new Error(`Batch size is 0 or NaN. Please choose a non-zero fraction.`);
      }
      this.model.fit(this.datasetController.savedX, this.datasetController.savedY, {
        batchSize,
        epochs: Config.NUM_EPOCHS,
        callbacks: {
          onBatchEnd: async (batch, logs) => observer.next(logs.loss),
        }
      });
    });
  }

  public async predict() {
    let isPredicting = true;

    while (isPredicting) {
      const predictedClass = tf.tidy(() => {
        const img = this.webcam.capture();
        const embeddings = this.truncatedMobileNet.predict(img);
        const predictions: Tensor<Rank> | Tensor<Rank>[] = this.model.predict(embeddings);
        return (predictions as Tensor<Rank>).as1D().argMax();
      });

      const classId = (await predictedClass.data())[0];
      predictedClass.dispose();
      console.log(classId);
      await tf.nextFrame();
    }

    isPredicting = false;
  }

  public setExampleHandler(label): void {
    tf.tidy(() => {
      const img: Tensor = this.webcam.capture();
      this.datasetController.addExample(this.truncatedMobileNet.predict(img) as Tensor, label);
    });
  }

  private getModel(): Model {
    return tf.sequential({
      layers: [
        tf.layers.flatten({
          inputShape: this.truncatedMobileNet.outputs[0].shape.slice(1)
        }),
        tf.layers.dense({
          units: Config.NUM_HIDDEN_LAYERS,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
          useBias: true
        }),
        tf.layers.dense({
          units: Config.NUM_CLASSES,
          kernelInitializer: 'varianceScaling',
          useBias: false,
          activation: 'softmax'
        })
      ]
    });
  }
}
