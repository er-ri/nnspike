    def high_speed_avoid(self, image: np.ndarray) -> Tuple[Optional[float], Optional[SpeedTuple], Mode]:

        if not self._init:
            self.initialize_action(motor_side=self.course)
        phase = self._phase
        status = self._status

        # phase0: 黄色検出＆中央追従。一定量検出で次フェーズへ。未検出時は中央追従・HIGH_SPEED_AVOID返却
        if phase.get_phase() == 0:
            print(f"[DEBUG] phase=0 yellow_pixel_count={find_bottle_center(image=image, color='yellow')[2]}")
            _, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow")
            if yellow_pixel_count > 3000:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, 100), Mode.HIGH_SPEED_AVOID

        # phase1: 黄色領域判定。一定量検出で次フェーズへ。未満なら中心または中央追従・HIGH_SPEED_AVOID返却
        if phase.get_phase() == 1:
            yellow_cx, _, yellow_pixel_count = find_bottle_center(image=image, color="yellow")
            print(f"[DEBUG] phase=1 yellow_pixel_count={yellow_pixel_count} yellow_cx={yellow_cx}")
            if yellow_pixel_count > 14000:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                if yellow_cx is not None:
                    target_x = yellow_cx[0]
                else:
                    target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, 70), Mode.HIGH_SPEED_AVOID

        # phase2: 右モーター位置差分が閾値未満なら旋回。閾値以上で次フェーズへ
        if phase.get_phase() == 2:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            print(f"[DEBUG] phase=2 position_diff={position_diff}")
            if position_diff < 250:
                if self.course == "right":
                    return None, (0, 70, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (70, 0, 0), Mode.HIGH_SPEED_AVOID
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase3: 黒水平ライン検出まで中央追従。未検出時は中央追従、検出でphase4へ、右モーター位置記録
        if phase.get_phase() == 3:
            print(f"[DEBUG] phase=3 intersection_detected={is_lower_horizontal_line_detected(image, intersection_y=450, roi=ROI_LINE_HORIZON3)}")
            if is_lower_horizontal_line_detected(image, intersection_y=450, roi=ROI_LINE_HORIZON3):
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                if self.course == "right":
                    return None, (70, 50, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (50, 70, 0), Mode.HIGH_SPEED_AVOID

        # phase4: 右モーター移動距離が閾値未満なら中央追従、閾値以上でphase5へ、右モーター位置記録
        if phase.get_phase() == 4:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            print(f"[DEBUG] phase=4 position_diff={position_diff}")
            if position_diff < 100:
                return None, (BASE_SPEED, BASE_SPEED, 0), Mode.HIGH_SPEED_AVOID
            else:
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase5: 左旋回。距離が閾値未満なら旋回継続、閾値以上で次フェーズへ。閾値未満かつ垂直黒ライン検出で次フェーズへ
        if phase.get_phase() == 5:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            print(f"[DEBUG] phase=5 position_diff={position_diff}")
            if position_diff < 300 and not is_vertical_black_line_detected(image, roi=ROI_LOOP, center_tolerance=120):
                if self.course == "right":
                    return None, (0, BASE_SPEED, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (BASE_SPEED, 0, 0), Mode.HIGH_SPEED_AVOID
            phase.next_phase()
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        if phase.get_phase() == 6:
            print(f"[DEBUG] phase=0 yellow_pixel_count={find_bottle_center(image=image, color='yellow')[2]}")
            if is_fast_corner_detected(image):
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, 70), Mode.HIGH_SPEED_AVOID

        if phase.get_phase() == 7:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            print(f"[DEBUG] phase=5 position_diff={position_diff}")
            if position_diff < 300 and not is_vertical_black_line_detected(image, roi=ROI_LOOP, center_tolerance=120):
                if self.course == "right":
                    return None, (40, 70, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (70, 40, 0), Mode.HIGH_SPEED_AVOID
            phase.next_phase()
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        if phase.get_phase() == 8:
            print(f"[DEBUG] phase=0 yellow_pixel_count={find_bottle_center(image=image, color='yellow')[2]}")
            if is_fast_corner_detected(image):
                phase.next_phase()
                phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))
            else:
                target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
                return target_x, (0, 0, 100), Mode.HIGH_SPEED_AVOID

        if phase.get_phase() == 9:
            position_start = phase.get_position_start("position_start")
            current_pos = self.get_motor_position(self.course, status=status)
            position_diff = abs(current_pos - position_start)
            print(f"[DEBUG] phase=5 position_diff={position_diff}")
            if position_diff < 300 and not is_vertical_black_line_detected(image, roi=ROI_LOOP, center_tolerance=120):
                if self.course == "right":
                    return None, (40, 70, 0), Mode.HIGH_SPEED_AVOID
                else:
                    return None, (70, 40, 0), Mode.HIGH_SPEED_AVOID
            phase.next_phase()
            phase.set_position_start("position_start", self.get_motor_position(self.course, status=status))

        # phase6: 状態リセットし右端/左端追従モードへ復帰
        if phase.get_phase() == 10:
            target_x = self.get_target_x_by_course(image, OFFSET_Y, self.course)
            blue_area = get_blue_line_pixel(image)
            print(f"[DEBUG] phase=6 blue_area={blue_area}")
            if blue_area > BLUE_AREA_MAX_THRESHOLD:
                self.reset_action()
                return target_x, None, Mode.DOUBLE_LOOP
            else:
                return target_x, None, Mode.HIGH_SPEED_AVOID

        print("[avoid_obstacle] Unexpected state reached.")
        return None, None, Mode.HIGH_SPEED_AVOID