import * as tf from '@tensorflow/tfjs';
import { Model, Rank, Tensor } from '@tensorflow/tfjs';
import { WebCamera } from './webcamera';
import { DatasetController } from './dataset-controller';
import { Config } from './config';
import { BehaviorSubject, Observable, Observer } from 'rxjs';
import { switchMapTo } from 'rxjs/operators';

export class NetController {

  public webcam: WebCamera;

  private datasetController = new DatasetController(Config.NUM_CLASSES);
  private truncatedMobileNet: Model;
  private model: Model;

  constructor(private video: HTMLVideoElement) {
    this.webcam = new WebCamera(video);
  }

  public init(): void {
    this.webcam.setup()
      .pipe(
        switchMapTo(this.loadTruncatedMobileNet$()),
      )
      .subscribe((model: Model) => {
        this.truncatedMobileNet = model;
        tf.tidy(() => this.truncatedMobileNet.predict(this.webcam.capture()));
      });
  }

  public train$(): Observable<number> {
    return Observable.create((observer: Observer<number>) => {
      this.model = this.getModel();
      const optimizer = tf.train.adam(Config.LEARNING_RATE);

      this.model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
      this.actualModelTrain(observer);
    });
  }

  public predictFromCameraCapture$(): Observable<number> {
    return Observable.create(observer => {
      const predict = () => {
        const predictedClass = tf.tidy(() => {
          const img = this.webcam.capture();
          const embeddings = this.truncatedMobileNet.predict(img);
          const predictions: Tensor<Rank> | Tensor<Rank>[] = this.model.predict(embeddings);
          return (predictions as Tensor<Rank>).as1D().argMax();
        });

        predictedClass.data()
          .then(
            (data: Int32Array) => {
              observer.next(data[0]);
              predictedClass.dispose();

              return tf.nextFrame();
            }
          )
          .then(() => predict());
      };
      predict();
    });
  }

  public setExampleHandler(label: number): void {
    tf.tidy(() => {
      const img: Tensor = this.webcam.capture();
      this.datasetController.addExample(this.truncatedMobileNet.predict(img) as Tensor, label);
    });
  }

  private loadTruncatedMobileNet$(): Observable<Model> {
    return Observable.create(observer => {
      tf.loadModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
      ).then(
        (mobilenet: Model) => {
          const layer = mobilenet.getLayer('conv_pw_13_relu');
          observer.next(tf.model({ inputs: mobilenet.inputs, outputs: layer.output }));
          observer.complete();
        }
      );
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

  private actualModelTrain(observer: Observer<number>): void {
    const batchSize = Math.floor(this.datasetController.savedX.shape[0]);

    this.model.fit(this.datasetController.savedX, this.datasetController.savedY, {
      batchSize,
      epochs: Config.NUM_EPOCHS,
      callbacks: {
        onBatchEnd: async (batch, logs) => observer.next(logs.loss),
      }
    })
    .then(() => observer.complete());
  }
}
