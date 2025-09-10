
from constants import BASE_SPEED


def avoid_obstacle(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:
        """
        障害物回避モードの制御。
        各フェーズで以下の処理を行う:
        
        - phase0: 黄色領域検出で次フェーズへ。未検出時はget_target_x_by_course_safeでAVOID_OBSTACLE返却。
        - phase1: 黄色領域検出で次フェーズへ。未検出時は黄色重心またはget_target_x_by_course_safeでAVOID_OBSTACLE返却。
        - phase2: 左旋回。所定距離未満は左右速度調整、到達で次フェーズへ。
        - phase3: 直進。所定距離未満は定速走行、到達で次フェーズへ。
        - phase4: intersection_y付近で黒水平ライン検出まで定速走行。所定距離未満は定速走行、以上で黒ライン検出判定。検出で次フェーズへ。
        - phase5: 右モーター移動距離が所定値未満なら定速走行、以上で次フェーズへ。
        - phase6: 左旋回。短距離は低速、長距離は高速。一定距離未満かつ垂直黒ライン検出で次フェーズへ。
        - phase7: コーナー検出で次フェーズへ。未検出時はget_target_x_by_courseでAVOID_OBSTACLE返却。
        - phase8: 右モーターが所定距離進行後、垂直黒ライン判定で次フェーズへ。
        - phase9: コーナー検出で状態リセット、他はAVOID_OBSTACLE継続。
        - phase10: 右モーターが所定距離進行後、垂直黒ライン判定で次フェーズへ。
        - phase11: 青面積判定または右モーターが所定距離進行でDOUBLE_LOOP、そうでなければAVOID_OBSTACLE継続。
        
        各フェーズで条件に応じて速度・モード・ターゲット座標を返却する。
        """

        if not self._init:
            self.initialize_action(motor_side=self.course)
            self.pid.Kp = 0.1  # 🏆 36回段階テスト結果：バランス0.624で最適（効率0.543 + 制御力0.590）
            self.pid.Ki = 0
            self.pid.Kd = 0.1  # 安定した微分制御で自然安定性向上
            self.pid.output_limits = (-1, 1)  # テスト結果による最適制御範囲

        phase = self._phase
        status = self._status

        # phase0: 領域検出で次フェーズへ。未検出時は中央追従・回避モード返却
        if phase.get_phase() == 0:
            _, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow", roi=ROI_COLOR)
            if yellow_pixel_count > 5000:
                print(f"[DEBUG] phase0→phase1: yellow_pixel_count={yellow_pixel_count} > 5000")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course_safe(image, self.opposite_course)
                return target_x, (0, 0, HIGH_SPEED_BASE), Mode.AVOID_OBSTACLE

        # phase1: 領域検出で次フェーズへ。未検出時は中心または中央追従・回避モード返却
        if phase.get_phase() == 1:
            yellow_cx, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow", roi=ROI_COLOR)
            if yellow_pixel_count > 18000:
                print(f"[DEBUG] phase1→phase2: yellow_pixel_count={yellow_pixel_count} yellow_cx={yellow_cx} > 18000")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                if yellow_cx is not None:
                    target_x = yellow_cx[0]
                else:
                    target_x = self.get_target_x_by_course_safe(image, self.opposite_course)
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        # phase2: 左旋回（条件成立まで速度調整、到達で次フェーズへ・モーター位置記録）
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 350:
                if self.course == "right":
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
            else:
                print(f"[DEBUG] phase2→phase3: position_diff={position_diff} current_pos={current_pos} >= 400")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase3: 直進（条件成立まで定速走行、到達で次フェーズへ・モーター位置記録）
        if phase.get_phase() == 3:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 200:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.AVOID_OBSTACLE
            else:
                print(f"[DEBUG] phase3→phase4: position_diff={position_diff} current_pos={current_pos} >= 200")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.opposite_course, status=status))

        # phase4: 条件成立まで定速走行、成立で判定・次フェーズへ（モーター位置記録）
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.opposite_course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 200:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE
            if is_lower_horizontal_line_detected(image, intersection_y=450, roi=ROI_LINE_HORIZON3):
                print(f"[DEBUG] phase4→phase5: position_diff={position_diff} current_pos={current_pos} (horizontal line detected)")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                if self.course == "right":
                    return None, (70, 40, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (40, 70, 0), Mode.AVOID_OBSTACLE

        # phase5: 右モーター移動距離が所定値未満なら中央追従、以上で次フェーズへ、右モーター位置記録。
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff < 200:
               return None, (BASE_SPEED, BASE_SPEED, 0), Mode.AVOID_OBSTACLE
            else:
                print(f"[DEBUG] phase5→phase6: position_diff={position_diff} current_pos={current_pos}")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase6: 左旋回（短距離は低速、長距離は高速、所定値以上で次フェーズへ。所定値未満かつ垂直黒ライン検出で次フェーズへ）
        if phase.get_phase() == 6:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            distance = abs(current_pos - position_start)
            if distance < 200:
                if self.course == "right":
                    return None, (5, 35, 0), Mode.AVOID_OBSTACLE
                else:
                    return None, (35, 5, 0), Mode.AVOID_OBSTACLE
            elif distance < 500:
                vertical_detected = is_vertical_black_line_detected(image, roi=ROI_LOOP, center_tolerance=150)
                if vertical_detected:
                    print(f"[DEBUG] phase6→phase7: distance={distance} current_pos={current_pos} (vertical black line detected)")
                    phase.next_phase()
                    phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
                    self.pid.Kp = 50
                    self.pid.Ki = 0
                    self.pid.Kd = 5
                    self.pid.output_limits = (-BASE_SPEED, BASE_SPEED)
                    return None, None, Mode.AVOID_OBSTACLE
                else:
                    if self.course == "right":
                        return None, (5, 35, 0), Mode.AVOID_OBSTACLE
                    else:
                        return None, (35, 5, 0), Mode.AVOID_OBSTACLE
            else:
                print(f"[DEBUG] phase6→phase7: distance={distance} current_pos={current_pos} >= 500")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
                self.pid.Kp = 50
                self.pid.Ki = 0
                self.pid.Kd = 5
                self.pid.output_limits = (-BASE_SPEED, BASE_SPEED)
                return None, None, Mode.AVOID_OBSTACLE

        # phase7: 所定距離進行で次フェーズへ。未到達時はget_target_x_by_courseでAVOID_OBSTACLE返却
        if phase.get_phase() == 7:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff >= 400:
                print(f"[DEBUG] phase7→phase8: コーナー検出: position_diff={position_diff} current_pos={current_pos} >= 400 → phase8移行")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        # phase8: 右モーターが所定距離進行後、垂直黒ライン判定で次フェーズへ。未到達時はget_target_x_by_courseでAVOID_OBSTACLE返却
        if phase.get_phase() == 8:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            if position_diff > 2900:
                print(f"[DEBUG] phase8→phase9: 右モーター距離: position_diff={position_diff} current_pos={current_pos} >= 2900 → phase9移行")
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.opposite_course)
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        # phase9: 青面積判定または右モーターが所定距離進行でDOUBLE_LOOP、そうでなければAVOID_OBSTACLE継続
        if phase.get_phase() == 9:
            blue_area = get_blue_line_pixel(image)
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            if blue_area > BLUE_AREA_MAX_THRESHOLD or position_diff >= 200:
                print(f"[DEBUG] phase9→DOUBLE_LOOP: 青面積または右モーター距離: position_diff={position_diff} blue_area={blue_area} current_pos={current_pos} > threshold → DOUBLE_LOOP移行")
                self.reset_action()
                return target_x, None, Mode.DOUBLE_LOOP
            else:
                return target_x, (0, 0, BASE_SPEED), Mode.AVOID_OBSTACLE

        print("[avoid_obstacle] Unexpected state reached.")
        return None, None, Mode.AVOID_OBSTACLE
