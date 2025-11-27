import os
from tqdm import tqdm

def distribute_commands(input_file="command.txt", output_files=["run45.sh", "run55.sh", "run65.sh", "run75.sh"]):
    """
    command.txt 파일을 읽어 각 줄을 줄 번호 % 4의 결과에 따라 
    output_files에 분배합니다.
    """
    
    print(f"입력 파일: {input_file}")
    print(f"출력 파일: {output_files}")

    try:
        # 파일이 비어있는지 확인하고 새로 시작합니다. (코드 실행 전 초기화)
        for outfile in output_files:
            if os.path.exists(outfile):
                 with open(outfile, 'w') as f:
                    f.write("") # 파일 내용 초기화
            else:
                # 파일이 존재하지 않으면, 다음 코드가 파일 생성 후 내용을 쓸 것입니다.
                pass 

        total_lines = 0
        with open(input_file, 'r') as f:
            total_lines = sum(1 for line in f)

        print(f"총 명령어 수: {total_lines}개")
                
        # command.txt 파일 열기
        with open(input_file, 'r') as infile:
            for line_number, command in tqdm(enumerate(infile), total=total_lines, desc="명령어 분배 진행"):
                file_index = line_number % 4
                output_file = output_files[file_index]
                with open(output_file, 'a') as outfile:
                    outfile.write(command.strip() + "\n")

        print("\n✅ 명령어 분배가 완료되었습니다.")

    except FileNotFoundError:
        print(f"\n❌ 오류: 입력 파일 '{input_file}'을(를) 찾을 수 없습니다.")
    except Exception as e:
        print(f"\n❌ 처리 중 오류가 발생했습니다: {e}")

distribute_commands()
