import { Component, ViewChild, OnInit, ElementRef, AfterViewInit } from '@angular/core';
import { WebCamera } from './webcamera';
import { NetController } from './net-controller';
import { Tensor } from '@tensorflow/tfjs';
import { interval, BehaviorSubject } from 'rxjs';
import { take, finalize } from 'rxjs/operators';
import { Config } from './config';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, AfterViewInit {
  @ViewChild('video') videoElement: ElementRef<HTMLVideoElement>;
  @ViewChild('stoneThumb') stoneThumbCanvasElement: ElementRef<HTMLCanvasElement>;
  @ViewChild('scissorsThumb') scissorsThumbCanvasElement: ElementRef<HTMLCanvasElement>;
  @ViewChild('paperThumb') paperThumbCanvasElement: ElementRef<HTMLCanvasElement>;

  public isStoneCaptured = false;
  public isScissorsCaptured = false;
  public isPaperCaptured = false;
  public isButtonLocked = false;
  public errorLoss: string = null;
  public isTrained = false;
  public isGameStarted = false;

  private netController: NetController;

  ngOnInit(): void {}

  ngAfterViewInit(): void {
    this.netController = new NetController(this.videoElement.nativeElement);
  }

  public onStone(): void {
    this.isStoneCaptured = true;
    this.handleCamera(
      () => {
        this.netController.setExampleHandler(0);
        this.drawThumb(this.stoneThumbCanvasElement.nativeElement);
      },
    );
  }

  public onScissors(): void {
    this.isScissorsCaptured = true;
    this.handleCamera(
      () => {
        this.netController.setExampleHandler(1);
        this.drawThumb(this.scissorsThumbCanvasElement.nativeElement);
      },
    );
  }

  public onPaper(): void {
    this.isPaperCaptured = true;
    this.handleCamera(
      () => {
        this.netController.setExampleHandler(2);
        this.drawThumb(this.paperThumbCanvasElement.nativeElement);
      },
    );
  }

  public onTrain(): void {
    this.netController.train$()
      .subscribe(
        (errorLoss: number) => {
          this.errorLoss = errorLoss.toFixed(5);
          this.isTrained = (errorLoss < Config.LEARNING_RATE);
        },
      );
  }

  public onPredict(): void {
    this.netController.predict();
  }

  private handleCamera(callback: Function, finalizeCallback?: Function): void {
    this.isButtonLocked = true;
    interval(100)
      .pipe(
        take(50),
        finalize(() => {
          this.isButtonLocked = false;
          return finalizeCallback && finalizeCallback();
        }),
      )
      .subscribe(() => callback());
  }

  private drawThumb(canvas: HTMLCanvasElement) {
    const image: Tensor = this.netController.webcam.capture();
    const [width, height] = [224, 224];
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
      imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
      imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }

}
