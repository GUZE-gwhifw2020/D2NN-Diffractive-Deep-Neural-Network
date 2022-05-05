Dim fileName As String
Dim sline As String
Dim iLay,ii,jj,i0 As Integer
Dim sPara As Double
Dim sss
Dim N As Integer      ' 单元数
Dim D As Double       ' 距离

WCS.ActivateWCS "local"

fileName = "text.txt"

Open fileName For Input As #1
	Do While Not EOF(1)
		Line Input #1,sline
		Debug.Print sline
		
		i0 = 0
		IF(instr(1, sline, "#") <> 0) THEN
			' 存在#字符，说明是数据段
			
			' Split拆分
			' Example: sline = "#2#28#100.0#-1.2 1.3 0.1 0.1"
			sss = Split(sline, "#")

			' 第一段层数
			iLay = Val(sss(1))
			
			' 第二段单元数
			N = Val(sss(2))
			
			' 第三段距离
			D = Val(sss(3))
			
			' 第四段结构(相位)段
			sss = Split(sss(4), " ")

			WCS.MoveWCS "local", "0.0", "0.0", D
			For ii = 1 To N
			For jj = 1 To N
				sPara = Val(sss(i0))
				With Brick
					.Reset 
					.Name "Brick" & Trim(str(iLay)) & "_" & Trim(str(i0)) 
					.Component "component1" 
					.Material "PEC" 
					.Xrange sPara + pi*(2*ii-1), -sPara + pi*(2*ii-1)
					.Yrange sPara + pi*(2*jj-1), -sPara + pi*(2*jj-1)
					.Zrange "0", "1" 
					.Create
				End With
				i0 = i0 + 1
			Next jj
			Next ii
			WCS.AlignWCSWithGlobalCoordinates 
		END IF
	Loop
Close #1

WCS.ActivateWCS "global"
