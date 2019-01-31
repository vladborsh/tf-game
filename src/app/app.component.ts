import { Component, ViewChild, OnInit, ElementRef, AfterViewInit } from '@angular/core';
import { WebCamera } from './webcamera';
import { NetController } from './net-controller';
import { Tensor } from '@tensorflow/tfjs';
import { interval, BehaviorSubject, timer, Observable, Subject, combineLatest } from 'rxjs';
import { take, finalize, tap, map, last, takeUntil } from 'rxjs/operators';
import { Config } from './config';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements AfterViewInit {
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
  public predicted: number;
  public humanTurn: number;
  public computerTurn: number;
  public timer$: Observable<number>;
  public score = { human: 0, computer: 0 };

  private netController: NetController;
  private timerFinished$ = new Subject<void>();
  private finalHumanTurn$ = new Subject<number>();
  private finalComputerTurn$ = new Subject<number>();

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
          this.isTrained = (errorLoss < 0.001);
        },
      );
  }

  public onStart(): void {
    this.isGameStarted = true;
    this.startTimer();
  }

  public startTimer(): void {
    this.timer$ = interval(1000)
      .pipe(
        take(Config.TIMER + 1),
        map(val => Config.TIMER - val),
        finalize(() => {
          this.finalComputerTurn$.next(Math.floor(Math.random() * 3));
          this.timerFinished$.next();
        })
      );

    combineLatest(this.netController.predictFromCameraCapture$(), this.finalComputerTurn$)
      .pipe(
        take(1),
      )
      .subscribe(
        ([finalHumanTurn, finalComputerTurn]: [number, number]) => {
          this.humanTurn = finalHumanTurn;
          this.computerTurn = finalComputerTurn;
          const win = this.compare(finalHumanTurn, finalComputerTurn);
          this.score = {
            human: win === 1 ? ++this.score.human : this.score.human,
            computer: win === -1 ? ++this.score.computer : this.score.computer,
          };
          this.startTimer();
        }
      );
  }

  private compare(turn1: number, turn2: number): number {
    return (turn1 === turn2)
      ? 0
      : (turn1 === 0 && turn2 === 1) || (turn1 === 1 && turn2 === 2) || (turn1 === 2 && turn2 === 0)
        ? 1
        : -1;
  }

  private handleCamera(callback: Function, finalizeCallback?: Function): void {
    this.isButtonLocked = true;
    interval(Config.CAMERA_CAPTURE_INTERVAL)
      .pipe(
        take(Config.CAMERA_CAPTURE_SET_SIZE),
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
